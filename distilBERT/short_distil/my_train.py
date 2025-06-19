import pickle
import torch
import numpy as np

from my_distiller import Distiller
from lm_seqs_dataset import LmSeqsDataset
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DistilBertForMaskedLM,
    DistilBertConfig,
)


class Params:
    dump_path = "./output_distillation"
    n_epoch = 3
    batch_size = 8
    gradient_accumulation_steps = 1
    learning_rate = 5e-4
    adam_epsilon = 1e-6
    warmup_prop = 0.05
    weight_decay = 0.01
    max_grad_norm = 5.0
    log_interval = 100
    checkpoint_interval = 500

    # MLM setup
    mlm = True
    mlm_mask_prop = 0.15
    word_mask = 0.8
    word_keep = 0.1
    word_rand = 0.1
    mlm_smoothing = 0.7

    # Loss weights
    temperature = 2.0   # for softmax smoothing
    alpha_ce = 0.5  # distillation loss (KL divergence between teacher and student softmax outputs)
                    # compares: softmax(student logits) vs softmax(teacher logits)

    alpha_mlm = 0.5  # masked language modeling loss (standard cross-entropy)
                     # compares: student predictions vs ground truth labels at masked positions

    alpha_cos = 1.0  # cosine embedding loss (align internal representations)
                     # compares: last hidden layer of student vs teacher

    # Hardware
    n_gpu = 1 if torch.cuda.is_available() else 0
    group_by_size = False
    restrict_ce_to_mask = True  # distill only on masked tokens

    # Required by distiller
    special_tok_ids = {}
    max_model_input_size = 512


def main():
    params = Params()

    # --- Tokenizer ---
    """
    The dataset was already preprocessed and tokenized, so we don't need the tokenizer for encoding.
    We load the same tokenizer that was used during preprocessing and 
    extract special token IDs for padding, masked_word and unknown_token
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # besides the token id's of the input we have this special token-id's
    special_tok_ids = {
        "pad_token": tokenizer.pad_token_id,    # padding
        "mask_token": tokenizer.mask_token_id,  # masked word
        "unk_token": tokenizer.unk_token_id,    # unknown token
    }
    params.special_tok_ids = special_tok_ids

    # --- Load preprocessed data (must be tokenized and pickled) ---
    with open("data/tokenized_data.pkl", "rb") as f:
        data = pickle.load(f)

    """
    # --- Load token frequency counts (for token masking probability) ---
    # Common tokens should be masked less often, rare tokens more often since they carry more information - inverse frequency weighting
    
    Example:
        For example we want to mask 15% of the tokens in each sentence.
        Then for each token in the sequence, we decide whether to include it in the 15% based on its masking weight, which comes from token_probs.
        Tokens with higher weights in token_probs (i.e. rare tokens) are more likely to be chosen as part of the 15%.
    """
    with open("data/token_counts.pkl", "rb") as f:
        counts = pickle.load(f)             # load counts how often tokens occurred
    token_probs = np.maximum(counts, 1) ** -params.mlm_smoothing
    for idx in special_tok_ids.values():
        token_probs[idx] = 0.0          # avoid masking special tokens
    token_probs = torch.from_numpy(token_probs).float()

    # --- Dataset ---
    """
    LmSeqsDataset wraps the pre-tokenized data and prepares it for masked language model training.
    - clean the data:
        + removes sequences that are too short ( less than 11 tokens)
        + split sequences that are too long
        + removes sequences that contain more than 50% unknown tokens
        
    - supports batching with proper padding
        converts a list of variable-length token sequences into a padded, fixed-size batch of tensors for training.
    """

    dataset = LmSeqsDataset(params=params, data=data)

    # student: untrained DistilBERT
    student_config = DistilBertConfig()
    student_config.output_hidden_states = True
    student = DistilBertForMaskedLM(config=student_config)

    # teacher: pretrained BERT
    teacher = BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)

    # Move to GPU if available
    if params.n_gpu > 0:
        student.cuda()
        teacher.cuda()

    # --- Distiller ---
    distiller = Distiller(
        params=params,
        dataset=dataset,
        token_probs=token_probs,
        student=student,
        teacher=teacher,
    )

    distiller.train()
    print("Training complete.")


if __name__ == "__main__":
    main()
