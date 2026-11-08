# Sen1Floods11: random vs geographic hold-out (SAR-only U-Net)

```
flood_project/
  requirements.txt
  preprocessing.py   index, clean, resize, cache, splits, normalisation, Dataset
  model.py           U-Net + masked BCE/Dice loss
  metrics.py         pixel counts, IoU/F1/precision/recall, bootstrap CIs
  train.py           trains every (split x channels x seed) run, saves checkpoints
  evaluation.py      evaluates saved runs, drop calculation, tables, figures
```

## Data
```
pip install -r requirements.txt
mkdir -p data
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand    data/S1Hand
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand  data/LabelHand
```
(Verify the bucket path in the Sen1Floods11 GitHub README.)

## Run
```
python preprocessing.py --data_root data --holdouts Pakistan   # cache + chip/split counts
python train.py         --data_root data --holdouts Pakistan --seeds 0 1 2
python evaluation.py    --cache cache --runs runs --out results
```
Training is resumable: finished runs are skipped unless `--overwrite` is given.

## Outputs (results/)
results_per_run.csv, results_summary.csv, generalisation_drop_per_seed.csv,
generalisation_drop_summary.csv, per_image_scores.csv, fig_*.png, pred_*.png
