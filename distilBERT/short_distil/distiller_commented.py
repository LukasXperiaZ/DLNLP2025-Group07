# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The distiller to distil the student.
Adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""

import math
import os
import time

import psutil
import torch
from grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from lm_seqs_dataset import LmSeqsDataset       # a dataset class for language modeling sequences.
                                                # Input: It takes data — a list of tokenized sequences (already tokenized
                                                # does some preprocessing like
                                                # conversion of sequences into arrays, splitting long esquences
from torch import nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup
from utils import logger


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


class Distiller:

    """
    params: config options from command-line arguments.
    dataset: the preprocessed dataset.
    token_probs: optional probabilities for token masking.
                 the token_probs argument represents a tensor containing probabilities for each token in the vocabulary.
                 These probabilities are used to guide which tokens should be masked more often during training.
                 Why use a tenser for this?
                    Not all tokens are equally useful for learning.
                    Some tokens are very frequent (like "the", "is") and provide little learning signal.
                    Others are rare or semantically rich and help more during training.
                    The probabilitys are calculated from token frequency statistics in a corpus.

    student, teacher: the two models.
    """
    def __init__(
        self, params: dict, dataset: LmSeqsDataset, token_probs: torch.tensor, student: nn.Module, teacher: nn.Module
    ):
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.dump_path
        self.multi_gpu = params.multi_gpu
        self.fp16 = params.fp16     # mixed precision training - use 16bit float instead of 32bit float in some places

        self.student = student
        self.teacher = teacher

        self.student_config = student.config
        self.vocab_size = student.config.vocab_size   # number of unique tokens

        if params.n_gpu <= 1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)

        """
        # Smart batching:
        create batches with similar sequence length. Some sequences are long, others short. 
        Without grouping by size a lot of memory is wasted by padding.
        GroupedBatchSampler: Draws mini-batches where all sequences are from the same length group.
        """
        if params.group_by_size:
            groups = create_lengths_groups(lengths=dataset.lengths, k=params.max_model_input_size)
            sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.batch_size)
        else:
            sampler = BatchSampler(sampler=sampler, batch_size=params.batch_size, drop_last=False)

        self.dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=dataset.batch_sequences)

        """
        # Loss function configuration – weights for each distillation loss component
        """
        self.temperature = params.temperature       # softmax with temp. -> class probabilitys of teacher are not only 0 or 1
        assert self.temperature > 0.0               #                       instead smoother probability distribution
                                                    #                        e.g., [0.6, 0.3, 0.1] instead of [0.99, 0.009, 0.001].
                                                    #                       This alloweds the student to learn more from the teacher.

        self.alpha_ce = params.alpha_ce             # Cross-entropy loss (student vs. teacher soft labels)
                                                    # Loss between student predictions and teacher's softened output

        # masked language modeling vs causal language modeling
        self.alpha_mlm = params.alpha_mlm           # Masked Language Modeling loss (student vs. ground truth labels)
                                                    # Student's performance on classic MLM task (compare to real labels)

        self.alpha_clm = params.alpha_clm           # Causal Language Modeling loss
                                                    # Causal Language Modeling means predicting the next token based only on previous tokens.
                                                    # This distillation framework is general - it supports
                                                    # 	Masked Language Modeling (MLM) models like BERT: masks words in the middle of the sentence and uses bidirectional context
                                                    #	Causal Language Modeling (CLM) models like GPT: predicting the next token based only on previous tokens.
                                                    #    alpha_mlm for BERT-like models
                                                    #    alpha_clm for GPT-like models

                                                    # Comparing student and teacher internal representation
        self.alpha_mse = params.alpha_mse           # Mean Squared Error loss (on hidden states or attention maps)
                                                    # absolute difference between the teacher's and student's internal values.
                                                    # Encourage student to imitate teacher’s internal representations

        self.alpha_cos = params.alpha_cos           # Cosine Similarity loss (on hidden states)
                                                    # The angle between the student and teacher vectors (i.e., their direction in space).
                                                    # Use cosine similarity to match teacher/student hidden states

        """
        # Masked language modeling
        If MLM is enabled:
            - mask some input 
            - train the student to predict those masked tokens

        These control how masked tokens are handled:
            word_mask: Replace with [MASK]
            word_keep: Keep the original token
            word_rand: Replace with a random token
        """
        self.mlm = params.mlm
        if self.mlm:
            logger.info("Using MLM loss for LM step.")
            self.mlm_mask_prop = params.mlm_mask_prop       # What proportion of tokens should be masked in the input
            assert 0.0 <= self.mlm_mask_prop <= 1.0
            assert params.word_mask + params.word_keep + params.word_rand == 1.0

            # creates a tensor with the probabilities for mask, keep, or random replacement
            self.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])

            # move the tensors to GPU
            self.pred_probs = self.pred_probs.to(f"cuda:{params.local_rank}") if params.n_gpu > 0 else self.pred_probs
            self.token_probs = token_probs.to(f"cuda:{params.local_rank}") if params.n_gpu > 0 else token_probs

            if self.fp16:
                self.pred_probs = self.pred_probs.half()
                self.token_probs = self.token_probs.half()

        else: # causal language modeling loss - predict the next token, no masking needed
            logger.info("Using CLM loss for LM step.")

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_mlm = 0
        self.last_loss_clm = 0
        if self.alpha_mse > 0.0:
            self.last_loss_mse = 0
        if self.alpha_cos > 0.0:
            self.last_loss_cos = 0
        self.last_log = 0

        """
        Initialize loss functions
            Initialize loss functions used during distillation
            Set up all relevant loss functions based on selected distillation objectives
        """
        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")              # For soft-label distillation (teacher vs. student logits)
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)           # For actual token prediction (MLM or CLM)
        if self.alpha_mse > 0.0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")                 # aligns internal representations
        if self.alpha_cos > 0.0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean") #  aligns vector directions

        """
        # Optimizer Setup
        Compute how many actual optimizer updates we'll make over the whole training run. 
        We are accumulating gradients over multiple batches.
        It's used to properly configure learning rate scheduling and warmup.
        For example for techniques like:
            - Linear learning rate warmup (gradually increase LR from 0 to max)
            - Learning rate decay (slowly reduce LR as training progresses)
            
        Weight decay — a form of L2 regularization.
        Applies a penalty proportional to the square of the weights' magnitude.
        This discourages large weights and helps prevent overfitting.
        """
        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.dataloader)                 # One epoch = full pass through the training data.
        num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )

        no_decay = ["bias", "LayerNorm.weight"]                     # group parameters that should not get weight decay
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))

        # create optimizer
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(0.9, 0.98)
        )

        # setup learning rate scheduler
        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )

        #  Enable Mixed Precision Training
        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
            self.student, self.optimizer = amp.initialize(
                self.student, self.optimizer, opt_level=self.params.fp16_opt_level
            )
            self.teacher = self.teacher.half()

        # Set up distributed training using multiple GPUs
        if self.multi_gpu:
            if self.fp16:
                from apex.parallel import DistributedDataParallel

                logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(self.student)
            else:
                from torch.nn.parallel import DistributedDataParallel

                logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(
                    self.student,
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    find_unused_parameters=True,
                )

        # Initialize TensorBoard for logging and visualization of the training prozess
        self.is_master = params.is_master
        if self.is_master:
            logger.info("--- Initializing Tensorboard")
            self.tensorboard = SummaryWriter(log_dir=os.path.join(self.dump_path, "log", "train"))
            self.tensorboard.add_text(tag="config/training", text_string=str(self.params), global_step=0)
            self.tensorboard.add_text(tag="config/student", text_string=str(self.student_config), global_step=0)

    def prepare_batch_mlm(self, batch):
        """
        Prepares a batch for Causal Language Modeling:
        Prepare the batch: from the token_ids and the lengths, compute the attention mask and the masked label for MLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            mlm_labels: `torch.tensor(bs, seq_length)` - The masked language modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, lengths = batch
        token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        """
        # initialize attention mask
            binary mask that informs which are the padding tokens and which are actual tokens
            [[1, 1, 1, 1, 0, 0, 0],    <- real tokens = 4
            [1, 1, 1, 0, 0, 0, 0],    <- real tokens = 3
            [1, 1, 1, 1, 1, 1, 1]]    <- real tokens = 7
            
            1 means "attend to this token" (real content)
            0 means "ignore this token" (padding)
        """
        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]

        bs, max_seq_len = token_ids.size()
        mlm_labels = token_ids.new(token_ids.size()).copy_(token_ids)

        x_prob = self.token_probs[token_ids.flatten()]
        n_tgt = math.ceil(self.mlm_mask_prop * lengths.sum().item())
        tgt_ids = torch.multinomial(x_prob / x_prob.sum(), n_tgt, replacement=False)
        pred_mask = torch.zeros(
            bs * max_seq_len, dtype=torch.bool, device=token_ids.device
        )  # previously `dtype=torch.uint8`, cf pytorch 1.2.0 compatibility
        pred_mask[tgt_ids] = 1
        pred_mask = pred_mask.view(bs, max_seq_len)

        # Create a boolean mask (pred_mask) indicating which token positions will be predicted (masked).
        pred_mask[token_ids == self.params.special_tok_ids["pad_token"]] = 0

        # mask a number of words == 0 [8] (faster with fp16)
        if self.fp16:
            n1 = pred_mask.sum().item()
            if n1 > 8:
                pred_mask = pred_mask.view(-1)
                n2 = max(n1 % 8, 8 * (n1 // 8))
                if n2 != n1:
                    pred_mask[torch.nonzero(pred_mask).view(-1)[: n1 - n2]] = 0
                pred_mask = pred_mask.view(bs, max_seq_len)
                assert pred_mask.sum().item() % 8 == 0, pred_mask.sum().item()

        """
        # masking:
        Randomly choose:
            If a token 
                - is masked ([MASK]) - typically 80% of masked tokens
                - keep original token - typically 10%  of masked tokens
                - set a random token - typically 10% of masked tokens
        """
        _token_ids_real = token_ids[pred_mask]
        _token_ids_rand = _token_ids_real.clone().random_(self.vocab_size)
        _token_ids_mask = _token_ids_real.clone().fill_(self.params.special_tok_ids["mask_token"])
        probs = torch.multinomial(self.pred_probs, len(_token_ids_real), replacement=True)
        _token_ids = (
            _token_ids_mask * (probs == 0).long()
            + _token_ids_real * (probs == 1).long()
            + _token_ids_rand * (probs == 2).long()
        )

        # replace original tokens with masked, original or random tokens as specified before
        token_ids = token_ids.masked_scatter(pred_mask, _token_ids)

        # only masked tokens should be considered for the MLM loss
        # Tokens with label -100 are ignored by the CrossEntropyLoss(ignore_index=-100)
        mlm_labels[~pred_mask] = -100

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size

        """
        token_ids - The actual token IDs (integers representing words/subwords from the vocabulary)
        mlm_labels - This tensor tells the model which tokens it is supposed to predict — and which ones to ignore.
                     It starts as a copy of the original token_ids.
                     Then:
                        At positions that are not masked, we set the value to -100. This tells PyTorch's loss function to ignore those positions.   
                        At positions that are masked, we keep the original token ID, because that’s the correct answer we want the model to learn to predict.
        """
        return token_ids, attn_mask, mlm_labels

    def prepare_batch_clm(self, batch):
        """
        causal language modeling loss - predict the next token
        Prepare the batch: from the token_ids and the lengths, compute the attention mask and the labels for CLM.
        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            clm_labels: `torch.tensor(bs, seq_length)` - The causal language modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, lengths = batch

        # This makes sure all sequences are aligned (rounded to a multiple of 8 for efficient FP16 training).
        token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        # This creates a binary mask: 1 where the token is real, 0 where it’s padding
        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]
        clm_labels = token_ids.new(token_ids.size()).copy_(token_ids)
        clm_labels[~attn_mask] = -100

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size

        return token_ids, attn_mask, clm_labels

    def round_batch(self, x: torch.tensor, lengths: torch.tensor):
        """
        Pads the sequences in x to the nearest fixed size (like 32, 64...).
        Makes GPU batching more efficient and avoids variable-length tensors.
        ---------------------------------------
        For float16 only.
        Sub-sample sentences in a batch, and add padding, so that each dimension is a multiple of 8.

        Input:
        ------
            x: `torch.tensor(bs, seq_length)` - The token ids.
            lengths: `torch.tensor(bs, seq_length)` - The lengths of each of the sequence in the batch.

        Output:
        -------
            x:  `torch.tensor(new_bs, new_seq_length)` - The updated token ids.
            lengths: `torch.tensor(new_bs, new_seq_length)` - The updated lengths.
        """
        if not self.fp16 or len(lengths) < 8:
            return x, lengths

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[idx, :slen]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(1)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            if self.mlm:
                pad_id = self.params.special_tok_ids["pad_token"]
            else:
                pad_id = self.params.special_tok_ids["unk_token"]
            padding_tensor = torch.zeros(bs2, pad, dtype=torch.long, device=x.device).fill_(pad_id)
            x = torch.cat([x, padding_tensor], 1)
            assert x.size() == (bs2, ml2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths

    def train(self):
        """
        The real training loop.
        """
        if self.is_master:
            logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()   # Set the student model to training mode (activates dropout, etc.)
        self.teacher.eval()    # Set the teacher model to evaluation mode (no dropout, no weight updates)

        for _ in range(self.params.n_epoch):    # train the specified number of epochs
            if self.is_master:
                logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")
            if self.multi_gpu:             # Synchronizes all processes in distributed/multi-GPU setup before continuing.
                torch.distributed.barrier()

            # A tqdm progress bar to show batch progress during the epoch. Only shown for the main process.
            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                if self.params.n_gpu > 0:   # Move the batch to GPU if available
                    batch = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)

                #  prepare input batch
                if self.mlm:
                    token_ids, attn_mask, lm_labels = self.prepare_batch_mlm(batch=batch)
                else:
                    token_ids, attn_mask, lm_labels = self.prepare_batch_clm(batch=batch)

                """
                Train on the batch
                This does one training step (forward pass, compute losses, backward pass, optimizer update if needed).
                """
                self.step(input_ids=token_ids, attention_mask=attn_mask, lm_labels=lm_labels)

                # Update progress bar, show current and average loss
                iter_bar.update()
                iter_bar.set_postfix(
                    {"Last_loss": f"{self.last_loss:.2f}", "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}"}
                )
            iter_bar.close()


            # Call end_epoch() which resets stats, saves a checkpoint, logs to TensorBoard, etc.
            if self.is_master:
                logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            self.end_epoch()

        if self.is_master:
            logger.info("Save very last checkpoint as `pytorch_model.bin`.")
            self.save_checkpoint(checkpoint_name="pytorch_model.bin")
            logger.info("Training is finished")

    def step(self, input_ids: torch.tensor, attention_mask: torch.tensor, lm_labels: torch.tensor):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """

        # forward step for student and teacher model to comput the logits and hidden states for both models
        # Teacher is in eval mode, so no gradient tracking
        if self.mlm:    # Masked Language Modeling, uses an attention_mask
            student_outputs = self.student(
                input_ids=input_ids, attention_mask=attention_mask
            )  # (bs, seq_length, voc_size)
            with torch.no_grad():   # Causal Language Modeling (predict next token) - does not use an attention_mask
                teacher_outputs = self.teacher(
                    input_ids=input_ids, attention_mask=attention_mask
                )  # (bs, seq_length, voc_size)
        else:
            student_outputs = self.student(input_ids=input_ids, attention_mask=None)  # (bs, seq_length, voc_size)
            with torch.no_grad():
                teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=None)  # (bs, seq_length, voc_size)

        # extract logits and hidden states
        s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
        t_logits, t_hidden_states = teacher_outputs["logits"], teacher_outputs["hidden_states"]
        assert s_logits.size() == t_logits.size()

        """
        # prepare the student and teacher logits (s_logits, t_logits) for the distillation loss computation
        Not every token is relevant for loss calculation (especially in MLM mode where only some tokens are masked).
        So we mask out irrelevant positions (e.g., padding or non-masked tokens).
        """
        if self.params.restrict_ce_to_mask: # uses the MLM label tensor to only compare positions that were masked
            mask = (lm_labels > -1).unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
        else:   # include all non-padding tokens, not just the masked ones
            mask = attention_mask.unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)

        # Apply the mask to select logits only from relevant positions
        s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()

        # compute distillation loss - difference between teacher and student logits
        loss_ce = (
            self.ce_loss_fct(
                nn.functional.log_softmax(s_logits_slct / self.temperature, dim=-1),
                nn.functional.softmax(t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        loss = self.alpha_ce * loss_ce

        # Add other losses
        if self.alpha_mlm > 0.0:    # Masked Language Modeling Loss
                                    # CrossEntropyLoss - goal: Predict masked tokens correctly.
                                    # compares students predictions with the ground truth
            loss_mlm = self.lm_loss_fct(s_logits.view(-1, s_logits.size(-1)), lm_labels.view(-1))
            loss += self.alpha_mlm * loss_mlm

        # Compare students predictions of the next token with the
        # ground truth for Causal Languague Modeling (= predicting the next token)
        if self.alpha_clm > 0.0:
            shift_logits = s_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_clm = self.lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss += self.alpha_clm * loss_clm

        # Mean Squared Error Loss - Match the student logits to the teacher logits.
        # Penalizes large differences between student and teacher predictions (after masking), aligning probability distributions.
        if self.alpha_mse > 0.0:
            loss_mse = self.mse_loss_fct(s_logits_slct, t_logits_slct) / s_logits_slct.size(
                0
            )  # Reproducing batchmean reduction
            loss += self.alpha_mse * loss_mse

        # Cosine Similarity Loss
        # Align the direction of hidden state vectors (final layer) from student and teacher.
        # Encourages student to learn similar internal representations, not just output probabilities.
        # We are comparing the last hidden layer of the student and teacher model
        #   The final hidden states are what the model uses to make predictions — aligning them helps the student
        #   mimic how the teacher “thinks,” not just what it predicts.
        if self.alpha_cos > 0.0:

            # Both models return hidden states for all layers. [-1] extracts the last layer's output
            s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
            t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)
            mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states)  # (bs, seq_length, dim)
                                                                            # adds a third dimension to make the attention mask compatible with the hidden states
            assert s_hidden_states.size() == t_hidden_states.size()
            dim = s_hidden_states.size(-1)

            # use the mask to only select embeddings of real tokens
            s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
            s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
            t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
            t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

            """
            The cosine loss expects a target tensor of 1s to indicate:
            “These two vectors should be similar.”
            So you build a vector like [1, 1, 1, ..., 1] with the same length as the number of compared embeddings.
            """
            target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
            loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
            loss += self.alpha_cos * loss_cos

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.alpha_mlm > 0.0:
            self.last_loss_mlm = loss_mlm.item()
        if self.alpha_clm > 0.0:
            self.last_loss_clm = loss_clm.item()
        if self.alpha_mse > 0.0:
            self.last_loss_mse = loss_mse.item()
        if self.alpha_cos > 0.0:
            self.last_loss_cos = loss_cos.item()

        self.optimize(loss)

        self.n_sequences_epoch += input_ids.size(0)

    def optimize(self, loss):
        """
        Backpropagation & Optimizer Update
            It covers loss normalization, backward pass, optional accumulation, gradient clipping, optimizer + learning rate scheduling.

        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.multi_gpu:  # average the loss over the gpu's since each gpu has its own loss
            loss = loss.mean()

        # If accumulating gradients over multiple steps, divide the loss to keep scale consistent.
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        # backward pass
        if self.fp16:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self.iter()  # Increments internal training step counter
        if self.n_iter % self.params.gradient_accumulation_steps == 0:  # Only do an optimizer .step() after enough mini-batches if accumulating.

            # Clip gradients to avoid exploding gradients (cap their size).
            if self.fp16:
                nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)

            self.optimizer.step()       # update model weights
            self.optimizer.zero_grad()  # reset gradients for next iteration.
            self.scheduler.step()       # possibly reduce learning rate (e.g., for warmup/decay).

    def iter(self):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        self.n_iter += 1
        self.n_total_iter += 1

        if self.n_total_iter % self.params.log_interval == 0:
            self.log_tensorboard()
            self.last_log = time.time()
        if self.n_total_iter % self.params.checkpoint_interval == 0:
            self.save_checkpoint()

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """
        if not self.is_master:
            return

        for param_name, param in self.student.named_parameters():
            self.tensorboard.add_scalar(
                tag="parameter_mean/" + param_name, scalar_value=param.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="parameter_std/" + param_name, scalar_value=param.data.std(), global_step=self.n_total_iter
            )
            if param.grad is None:
                continue
            self.tensorboard.add_scalar(
                tag="grad_mean/" + param_name, scalar_value=param.grad.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="grad_std/" + param_name, scalar_value=param.grad.data.std(), global_step=self.n_total_iter
            )

        self.tensorboard.add_scalar(
            tag="losses/cum_avg_loss_epoch",
            scalar_value=self.total_loss_epoch / self.n_iter,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(tag="losses/loss", scalar_value=self.last_loss, global_step=self.n_total_iter)
        self.tensorboard.add_scalar(
            tag="losses/loss_ce", scalar_value=self.last_loss_ce, global_step=self.n_total_iter
        )
        if self.alpha_mlm > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_mlm", scalar_value=self.last_loss_mlm, global_step=self.n_total_iter
            )
        if self.alpha_clm > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_clm", scalar_value=self.last_loss_clm, global_step=self.n_total_iter
            )
        if self.alpha_mse > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_mse", scalar_value=self.last_loss_mse, global_step=self.n_total_iter
            )
        if self.alpha_cos > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_cos", scalar_value=self.last_loss_cos, global_step=self.n_total_iter
            )
        self.tensorboard.add_scalar(
            tag="learning_rate/lr", scalar_value=self.scheduler.get_lr()[0], global_step=self.n_total_iter
        )

        self.tensorboard.add_scalar(
            tag="global/memory_usage",
            scalar_value=psutil.virtual_memory()._asdict()["used"] / 1_000_000,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(
            tag="global/speed", scalar_value=time.time() - self.last_log, global_step=self.n_total_iter
        )

    def end_epoch(self):
        """
        Wraps up after an epoch:
        Calls log_tensorboard.
        May save model checkpoint or reset state.
        --------------------

        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        if self.is_master:
            self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")
            self.tensorboard.add_scalar(
                tag="epoch/loss", scalar_value=self.total_loss_epoch / self.n_iter, global_step=self.epoch
            )

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Saves the student model’s current weights to disk.
        -----------------------
        Save the current state. Only by the master process.
        """
        if not self.is_master:
            return
        mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
        mdl_to_save.config.save_pretrained(self.dump_path)
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))
