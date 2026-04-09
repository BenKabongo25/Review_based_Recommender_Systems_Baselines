# RMG

PyTorch runner for **Reviews Meet Graphs** adapted to the same I/O format as the other baselines in this repository.

## Inputs

`main.py` uses:

- `--dataset_csv`: CSV file containing at least `user_id`, `item_id`, `rating`, and `review` (or `review_text`)
- `--train_idx`, `--eval_idx`, `--test_idx`: `.npy` index files
- `--output_dir`: directory for checkpoints and result JSON

## Paper-aligned defaults

The default hyper-parameters match the experimental setting you provided:

- word embedding dim: `300`
- CNN filters: `150`
- CNN window size: `3`
- user/item id embedding dim: `100`
- dropout: `0.25`
- optimizer: Adam
- batch size: `20`

## Example

```bash
python main.py \
  --dataset_name Toys14 \
  --dataset_csv /data/common/RecommendationDatasets/StatementDatasets/Toys14/dataset_vC.csv \
  --train_idx /data/common/RecommendationDatasets/StatementDatasets/Toys14/train_idx.npy \
  --eval_idx /data/common/RecommendationDatasets/StatementDatasets/Toys14/eval_idx.npy \
  --test_idx /data/common/RecommendationDatasets/StatementDatasets/Toys14/test_idx.npy \
  --pretrained_emb_path /path/to/GoogleNews-vectors-negative300.bin.gz \
  --output_dir /data/common/RecommendationDatasets/StatementDatasets/exps/RMG/Toys14 \
  --seed 42
```
