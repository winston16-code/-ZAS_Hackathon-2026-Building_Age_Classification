# Trained models

| File | What it is |
|---|---|
| `stage1_transfer_base.joblib` | CatBoost transfer base trained on Madrid (TRANSFER feature set, per-city scaling) |
| `stage2_amsterdam_n{5,25,50,100,200}.joblib` | Stage-2 Amsterdam models built with `build_stage2(n, seed=0)` |
| `config.json` | Feature lists (RICH, TRANSFER), smoothing setting, list of broken scenes |

**Not included:** `stage1_madrid.joblib` (the Madrid LightGBM + XGBoost + Random Forest ensemble, 245 MB),
which exceeds GitHub's 100 MB file limit. Running `notebooks/Group2_Submission.ipynb` regenerates it
(about 20 minutes on a GPU laptop, about 40–60 minutes on a CPU).

The Stage-2 objects are instances of the `Stage2` / `Ensemble` classes defined in the notebook, so load them
inside a session where the notebook's code cells have been run.
