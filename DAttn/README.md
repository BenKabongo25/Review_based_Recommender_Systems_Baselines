# Baseline Runner

Standalone runner adapted from `Neu-Review-Rec` with unified inputs:
- `--dataset_name --dataset_csv --train_idx --eval_idx --test_idx --output_dir --seed`

Run:
```bash
python main.py --dataset_name Toys14 --dataset_csv /path/dataset.csv --train_idx /path/train_idx.npy --eval_idx /path/eval_idx.npy --test_idx /path/test_idx.npy --output_dir /path/out --seed 42
```
