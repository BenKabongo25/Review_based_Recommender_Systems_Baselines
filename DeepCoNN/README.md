# DeepCoNN

Paper: *Deep Cooperative Neural Networks (DeepCoNN), arXiv:1701.04783*.

This implementation keeps the paper setup (DeepCoNN + FM, RMSprop, MSE) and now supports loading pretrained embeddings directly from CLI.

## Supported embedding inputs
- `--pretrained_emb_path /path/GoogleNews-vectors-negative300.bin.gz` (word2vec bin/gz)
- `--pretrained_emb_path /path/glove.840B.300d.txt --pretrained_emb_format glove_txt` (GloVe text)
- `--pretrained_emb_path /path/w2v.npy --pretrained_emb_format npy` (already-built matrix)

Notes:
- `word_dim` must match the pretrained dimension (paper: 300).
- OOV words are initialized randomly in `[-1, 1]`.

## Run (paper defaults)
```bash
python main.py \
  --dataset_name Toys14 \
  --dataset_csv /path/dataset_vC.csv \
  --train_idx /path/train_idx.npy \
  --eval_idx /path/eval_idx.npy \
  --test_idx /path/test_idx.npy \
  --pretrained_emb_path /home/kabongo/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz \
  --output_dir /path/DeepCoNN \
  --seed 42
```
