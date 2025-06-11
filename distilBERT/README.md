How the training of DistilBERT_small was done:
===
0. We make use of the distillation research project on the huggingface [github](https://github.com/huggingface/transformers-research-projects/tree/main/distillation).
1. Preprocess a subset of the english wikipedia in `extract_wikipedia_data.ipynb`
    * We only add sentences from an article if it contains at least 3 words.
    * Since we do not use all articles, we sample them randomly.
2. Binarize the data with executing `python scripts/binarized_data.py --file_path data/dump.txt --tokenizer_type bert --tokenizer_name bert-base-uncased --dump_file data/binarized_text`
    * Note that we modified the binarization slightly to use only sequences with a minimum number of 12 tokens (since it is filtered later anyway in `train.py`).
3. Count the occurences of each token in the data (needed for masked language modeling loss): `python scripts/token_counts.py --data_file data/binarized_text.bert-base-uncased.pickle --token_counts_dump data/token_counts.bert-base-uncased.pickle --vocab_size 30522`
4. Specify a .json configuration of distilbert small in `distilbert-small-uncased.json`
6. Generate BERT checkpoint for later initializing the model weights of distilBERT: `python scripts/extract/distilbert.py`
5. Distil using the train script by executing `train.sh`


Facts:
---
* Started with 236.370 wikipedia articles (randomly sampled) containing a total of ~5 million sentences with at least 3 words.
* For training, used 5.355.344 sequences.

#### DistilBERT (6 Layers):
* Training took around 1h:30min per epoch.
* We trained for TODO epochs.

#### DistilBERT_small (4 layers):
* Training took around 1h:20min per epoch.
* We trained for TODO epochs.

Interesting observations:
---
* When training, the VRAM usage first goes up continuously. 
* We tried to find the cause of it, but didn't succeed. 
* However, interestingly, after a certain time, it suddenly goes down to about 10/24 GB and then again continuously up. We observed that this behaviour was repeated a few number of times. However, after some time, it stayed high. 
* The most likely cause of this is that pytorch caches some information. 
* We fixed it by calling `torch.cuda.empty_cache()` if there is less than 2GB of VRAM remaining, which frees cached VRAM and thus prevents the static increase.
* Furthermore, due to a `torch.OutOfMemoryError`, we executed `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the shell to allow the CUDA memory allocator to be more flexible when allocating memory. Furthermore, we reduced the batch size from 48 to 32.
* Finally, we also modified the code to work with the `--fp16` option. This option specifies whether to use 16-bit (mixed) precision instead of 32-bit for certain components. This allows us to reduce the required VRAM drastically.