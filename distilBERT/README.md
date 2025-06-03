How the training of DistilBERT_small was done:
===
1. Preprocess a subset of the english wikipedia in `extract_wikipedia_data.ipynb`
2. Binarize the data with executing `python scripts/binarized_data.py --file_path data/dump.txt --tokenizer_type bert --tokenizer_name bert-base-uncased --dump_file data/binarized_text`
    * Note that we modified the binarization slightly to truncate sentences to a maximum of 512 tokens, which is the maximum input size of BERT/distilBERT.
3. Count the occurences of each token in the data (needed for masked language modeling loss): `python scripts/token_counts.py --data_file data/binarized_text.bert-base-uncased.pickle --token_counts_dump data/token_counts.bert-base-uncased.pickle --vocab_size 30522`
4. Specify a .json configuration of distilbert small in `distilbert-small-uncased.json`
5. Distil using the train script by executing `train.sh`
6. TODO