from __future__ import annotations
import argparse, glob, pathlib, joblib, datetime
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
from .features import FeatureBuilder
from .utils import load_config, ensure_dir

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    paths = cfg['paths']

    test = pd.read_csv(paths['test_csv'])
    feat_stats_path = f"{paths['model_dir']}/feature_stats.joblib"
    feat = FeatureBuilder(joblib.load(feat_stats_path) if pathlib.Path(feat_stats_path).exists() else None)
    X_test = feat.transform(test.copy())

    cat_cols = X_test.select_dtypes(include=['object','category']).columns.tolist()
    cat_idx = [X_test.columns.get_loc(c) for c in cat_cols]
    test_pool = Pool(X_test, cat_features=cat_idx)

    # load all fold models
    model_paths = sorted(glob.glob(f"{paths['model_dir']}/catboost_fold*.cbm"))
    if not model_paths:
        raise FileNotFoundError("No models found in model_dir. Run training first.")
    probas = []
    for mp in model_paths:
        m = CatBoostClassifier()
        m.load_model(mp)
        probas.append(m.predict_proba(test_pool)[:,1])
    proba_mean = np.mean(probas, axis=0)

    thr = cfg['threshold'].get('default', 0.5)
    pred = (proba_mean >= thr).astype(int)

    ensure_dir(paths['submissions_dir'])
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{paths['submissions_dir']}/submission_catboost_thr{str(thr).replace('.','')}_{ts}.csv"
    # respect sample_submission id if exists
    if pathlib.Path(paths['sample_csv']).exists():
        sample = pd.read_csv(paths['sample_csv'])
        if 'fraud' in sample.columns:
            sample['fraud'] = pred
            sample.to_csv(out_path, index=False)
        else:
            # fallback: just write a single column
            pd.DataFrame({'fraud': pred}).to_csv(out_path, index=False)
    else:
        pd.DataFrame({'fraud': pred}).to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(args.config)
