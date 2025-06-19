import math
import os
import time

import torch
from lm_seqs_dataset import LmSeqsDataset       # a dataset class for language modeling sequences.
                                                # Input: It takes data — a list of tokenized sequences (already tokenized
                                                # does some preprocessing like
                                                # conversion of sequences into arrays, splitting long esquences
from torch import nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup
from utils import logger



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
        self,
        params: dict,
        dataset: LmSeqsDataset,
        token_probs: torch.tensor,
        student: nn.Module,
        teacher: nn.Module
    ):
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.dump_path

        self.student = student
        self.teacher = teacher

        self.student_config = student.config
        self.vocab_size = student.config.vocab_size   # number of unique tokens

        # Sample individual indices randomly from the dataset
        # selecting sequences randomly from the dataset.
        index_sampler = RandomSampler(dataset)

        # Group the randomly selected sequences into batches of fixed size
        # sampler is an iterable that yields lists of sequence indices, where each list represents a batch.
        sampler = BatchSampler(index_sampler, batch_size=params.batch_size, drop_last=False)

        """
        Create a PyTorch DataLoader that:
        - Uses the custom dataset (LmSeqsDataset)
        - Draws random mini-batches using the batch sampler
        - Applies the custom collate_fn (batch_sequences) to pad and batch the data
        """
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

                                                    # Comparing student and teacher internal representation
        self.alpha_mse = params.alpha_mse           # Mean Squared Error loss (on hidden states or attention maps)
                                                    # absolute difference between the teacher's and student's internal values.
                                                    # Encourage student to imitate teacher’s internal representations

        self.alpha_cos = params.alpha_cos           # Cosine Similarity loss (on hidden states)
                                                    # The angle between the student and teacher vectors (i.e., their direction in space).
                                                    # Use cosine similarity to match teacher/student hidden states

        """
        Configure Masked Language Modeling (MLM) behavior.

        If MLM is enabled:
        - Randomly select a subset of input tokens to be masked.
        - Train the student model to predict the original values of these masked tokens.

        For each selected token, one of the following actions is taken (based on defined probabilities):
        - word_mask: Replace the token with [MASK]
        - word_keep: Keep the original token unchanged
        - word_rand: Replace the token with a random token from the vocabulary
        """
        self.mlm = params.mlm
        logger.info("Using MLM loss for LM step.")
        self.mlm_mask_prop = params.mlm_mask_prop       # What proportion of tokens should be masked in the input
        assert 0.0 <= self.mlm_mask_prop <= 1.0
        assert params.word_mask + params.word_keep + params.word_rand == 1.0

        # creates a tensor with the probabilities for mask, keep, or random replacement
        # basically a list of three values [prop. to mask, prop. to keep, prop. to replace with random token]
        self.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])

        # move the tensors to GPU
        self.pred_probs = self.pred_probs.to(f"cuda:{params.local_rank}") if params.n_gpu > 0 else self.pred_probs
        self.token_probs = token_probs.to(f"cuda:{params.local_rank}") if params.n_gpu > 0 else token_probs

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_mlm = 0
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

        self.setup_optimizer(params, student)

        self.is_master = params.is_master

    def setup_optimizer(self, params, student):
        """
        Initializes the adam optimizer and learning rate scheduler.

        + Optimizer initialization:
            Compute how many actual optimizer updates we'll make over the
            whole training run based on the hyperparameter params.gradient_accumulation_steps.
            By default we do a gradient update every 50 batches

            parser.add_argument(
                "--gradient_accumulation_steps",
                type=int,
                default=50,
                help="Gradient accumulation for larger training batches.",
            )

        + Weight decay (L2 regularization) - to prevent overfitting

        + Learning rate scheduler
            - learning rate starts small,
            - gradually increases for the first few steps (warmup phase),
            - and then linearly decays.
            This stabilizes training and helps prevent divergence in early updates.
            At the start, model weights are random, gradients can be noisy, and large updates can destabilize learning.

            The warmup phase is configured using the --warmup_prop argument, which defines
            what proportion of the total training steps should be used for learning rate warmup.
        """
        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1

        self.num_steps_epoch = len(self.dataloader)  # One epoch = full pass through the training data.
        num_train_optimization_steps = (
                int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )
        no_decay = ["bias", "LayerNorm.weight"]  # group parameters that should not get weight decay
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

    def prepare_batch_mlm(self, batch):
        """
        Prepares a batch for Mask Language Modeling:
        - create the attention mask, real tokens vs padding, want to ignore padding tokens
        - mask some tokens
        - Setting up the MLM labels so the model knows which tokens to predict

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
        # token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        # assert token_ids.size(0) == lengths.size(0)

        """
        # initialize attention mask
            binary mask that informs which are the padding tokens and which are actual tokens
            [[1, 1, 1, 1, 0, 0, 0],    <- real tokens = 4
            [1, 1, 1, 0, 0, 0, 0],    <- real tokens = 3
            [1, 1, 1, 1, 1, 1, 1]]    <- real tokens = 7
            
            1 means "attend to this token" (real content)
            0 means "ignore this token" (padding)
            
            details:
                We are comparing:
                    lengths[:, None] .... a tensor of shape (batch_size, 1)
                    with torch.arange(...)  ... a 1D tensor of shape (max_seq_len,)
                
                PyTorch automatically broadcasts these to: batch_size, max_seq_len)
        """
        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]

        bs, max_seq_len = token_ids.size()          # extracts batch_size and max_seq_len from the shape of token_ids
        mlm_labels = token_ids.new(token_ids.size()).copy_(token_ids)  # copy the token ids of the batch

        """
        # select tokens to be masked
                + masking based on masking probability and based on how often a token occurs
                + learning to predict rare words is more valueable since these words give more information
                   token probabilitys:  max(count, 1) ^ (-alpha)       # see train.py for details
        """
        x_prob = self.token_probs[token_ids.flatten()]                 #  1D tensor of size (batch_size × seq_len). Each value represents the desirability (or probability) of masking that specific token.
        n_tgt = math.ceil(self.mlm_mask_prop * lengths.sum().item())   #  this computes the number of tokens that should be masked in total

        tgt_ids = torch.multinomial(x_prob / x_prob.sum(), n_tgt, replacement=False)    # normalize the token probabilities to turns x_prob into a valid probability distribution where the values sum to 1.
                                                                                        # then select tokens to be masked based on the probabilites
                                                                                        # e.g. mask the 17th, 24th, 89th, 103rd token in the flattened batch "token_ids"
        """
        # Create a boolean mask (pred_mask) indicating which token positions will be predicted (masked).
        pred_mask is a boolean matrix of shape (batch_size, seq_len) where:
            1 ... this token should be masked and predicted
            0 ... leave this token as-is
        """
        pred_mask = torch.zeros(
            bs * max_seq_len, dtype=torch.bool, device=token_ids.device
        )
        pred_mask[tgt_ids] = 1                              # sets entries at the sampled positions (tgt_ids) to True (1)
        pred_mask = pred_mask.view(bs, max_seq_len)         # reshape the flat boolean mask into the original (batch_size, seq_len) shape.
        pred_mask[token_ids == self.params.special_tok_ids["pad_token"]] = 0        # make sure padding tokens are not masked

        """
        # masking stategy:
            for each token selected for masking choose if a token 
            - is masked ([MASK]) - typically 80% of masked tokens
            - keep original token - typically 10%  of masked tokens
            - set a random token - typically 10% of masked tokens
        """
        _token_ids_real = token_ids[pred_mask]                                          # select the original token IDs at the positions we plan to mask
        _token_ids_rand = _token_ids_real.clone().random_(self.vocab_size)              # creates a random token for each masked position
        _token_ids_mask = _token_ids_real.clone().fill_(self.params.special_tok_ids["mask_token"])  # creates a version where all masked tokens are replaced with [MASK]
        probs = torch.multinomial(self.pred_probs, len(_token_ids_real), replacement=True)  # choose which tokens to mask, keep, randomize
        _token_ids = (                              # build the final masked token list by selecting the masked, original or randomized token based on the previous sampling
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

            # A tqdm progress bar to show batch progress during the epoch. Only shown for the main process.
            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                if self.params.n_gpu > 0:   # Move the batch to GPU if available
                    batch = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)

                token_ids, attn_mask, lm_labels = self.prepare_batch_mlm(batch=batch)

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


            # Call end_epoch() which resets stats, saves a checkpoint, does some logging, etc.
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

        # forward step for student and teacher model to compute the logits and hidden states for both models
        # Teacher is in eval mode, so no gradient tracking
        student_outputs = self.student(
            input_ids=input_ids, attention_mask=attention_mask
        )  # (bs, seq_length, voc_size)
        with torch.no_grad():   # Causal Language Modeling (predict next token) - does not use an attention_mask
            teacher_outputs = self.teacher(
                input_ids=input_ids, attention_mask=attention_mask
            )  # (bs, seq_length, voc_size)

        # extract logits and hidden states
        s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
        t_logits, t_hidden_states = teacher_outputs["logits"], teacher_outputs["hidden_states"]
        assert s_logits.size() == t_logits.size()

        """
        mask out irrelevant positions (e.g., padding or non-masked tokens)
            lm_labels > -1 ... returns a boolean tensor 
                               True where the value is greater than -1 (a masked token)
                               False where the value is -100 → not used in loss, ignore it
        """
        mask = (lm_labels > -1).unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)

        """
        # Use the mask to select logits only from relevant positions where mask == True
        """
        # selects only logits where mask == True for the student model
        s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask

        # selects only logits where mask == True for the teacher model
        t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()

        """
        # Loss calculation
        
        distillation loss
            To compare the probability distribution of the teachers soft-prediction and the students soft-predictions we use the Kullback-Leiber divergence.
            For this, we compute the softmax of the teacher's logits and the log-softmax of the student's logits.
            temperature scaling of logits to get a smoother probability distributions
            s_logits_slct / self.temperature
            t_logits_slct / self.temperature
        """
        # compute distillation loss - difference between teacher and student logits
        loss_ce = (
            self.ce_loss_fct(
                nn.functional.log_softmax(s_logits_slct / self.temperature, dim=-1),
                nn.functional.softmax(t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2      # temperature softening of the logits scales down the gradients,
                                           # therefore we multiply with a correcting factor T² see Hinton et al distillation paper (2015)
        )
        loss = self.alpha_ce * loss_ce

        # Add other losses
        if self.alpha_mlm > 0.0:    # Masked Language Modeling Loss
                                    # CrossEntropyLoss - goal: Predict masked tokens correctly.
                                    # compares students predictions with the ground truth
            loss_mlm = self.lm_loss_fct(s_logits.view(-1, s_logits.size(-1)), lm_labels.view(-1))
            loss += self.alpha_mlm * loss_mlm

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

            # Both models return hidden states for all layers. [-1] extracts the last layer's output, the prediction layer
            s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
            t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)

            # build the mask to ignore padding tokens
            mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states)  # (bs, seq_length, dim)
                                                                            # adds a third dimension to make the attention mask compatible with the hidden states
            assert s_hidden_states.size() == t_hidden_states.size()
            dim = s_hidden_states.size(-1)

            # use the mask to only select embeddings of real tokens
            # then reshape the tensor so we can compare the embeddings
            s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
            s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim) reshape into 2D tensor so we can compare the embeddings

            t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
            t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

            """
            # Cosine loss
            The cosine loss expects a target tensor of 1s to indicate:
            “These two vectors should be similar.”
            So you build a vector like [1, 1, 1, ..., 1] with the same length as the number of compared embeddings.
            
            target vector of shape (N,), where N is the number of selected (non-padding) tokens.
                +1 means: "These two vectors should be similar"
                -1 means: "These two vectors should be dissimilar"
            """
            target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)  creates a 1D tensor of shape (N,) where N is the number
                                                                                      #                     of selected (non-padding) tokens and Fills it with all 1s.
            loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
            loss += self.alpha_cos * loss_cos

        # track the total loss and the individual components of the loss (cross-entropy, MLM, MSE, cosine) for the current batch
        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.alpha_mlm > 0.0:
            self.last_loss_mlm = loss_mlm.item()
        if self.alpha_mse > 0.0:
            self.last_loss_mse = loss_mse.item()
        if self.alpha_cos > 0.0:
            self.last_loss_cos = loss_cos.item()

        self.optimize(loss)

        self.n_sequences_epoch += input_ids.size(0)     # track how many sequences were processed this epoch for logging

    def optimize(self, loss):
        """
        Backpropagation & Optimizer Update
            It covers loss normalization, backward pass, optional accumulation, gradient clipping, optimizer + learning rate scheduling.

        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        """
        # Check for NaN - could occur because of gradient explosion
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # If accumulating gradients over multiple steps, divide the loss to keep scale consistent.
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        # backward pass
        loss.backward()

        self.iter()  # Increments internal training step counter
        if self.n_iter % self.params.gradient_accumulation_steps == 0:  # Only do an optimizer .step() after enough mini-batches if accumulating.

            # Clip gradients to avoid exploding gradients (cap their size).
            nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)

            self.optimizer.step()       # update model weights
            self.optimizer.zero_grad()  # reset gradients for next iteration.
            self.scheduler.step()       # possibly reduce learning rate (e.g., for warmup/decay).

    def iter(self):
        """
        Update global counts, log progress, and save checkpoint.
        """
        self.n_iter += 1
        self.n_total_iter += 1

        if self.n_total_iter % self.params.log_interval == 0:
            self.log_progress()
            self.last_log = time.time()
        if self.n_total_iter % self.params.checkpoint_interval == 0:
            self.save_checkpoint()

    def log_progress(self):
        """
        Simple log message to show training progress.
        """
        if not self.is_master:
            return

        logger.info(
            f"Step {self.n_total_iter} | "
            f"Loss: {self.last_loss:.4f} | "
            f"CE: {self.last_loss_ce:.4f} | "
            f"MLM: {self.last_loss_mlm:.4f} | "
            f"MSE: {getattr(self, 'last_loss_mse', 0.0):.4f} | "
            f"COS: {getattr(self, 'last_loss_cos', 0.0):.4f} | "
            f"Avg loss: {self.total_loss_epoch / self.n_iter:.4f}"
        )

    def end_epoch(self):
        """
        Wraps up after an epoch:
        May save model checkpoint or reset state.
            stores the student model’s weights to disk
            This checkpoint system ensures we can:
                + recover from crashes
                + resume training
                + load a trained model later for inference or evaluation
        --------------------

        Finally arrived at the end of epoch (full pass on dataset).
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        if self.is_master:
            self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")

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
