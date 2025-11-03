from __future__ import annotations
import argparse, pathlib, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier, Pool

from .features import FeatureBuilder
from .utils import load_config, ensure_dir

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    paths = cfg['paths']
    train_csv = paths['train_csv']

    df = pd.read_csv(train_csv)
    y = df['fraud'] if 'fraud' in df.columns else df.iloc[:, -1]
    X = df.drop(columns=['fraud']) if 'fraud' in df.columns else df.iloc[:, :-1]

    feat = FeatureBuilder()
    X = feat.fit_transform(X)

    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    skf = StratifiedKFold(n_splits=cfg['cv']['n_splits'], shuffle=cfg['cv']['shuffle'], random_state=cfg['seed'])
    oof_proba = np.zeros(len(X), dtype=float)
    models = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        train_pool = Pool(X_trn, y_trn, cat_features=cat_idx)
        valid_pool = Pool(X_val, y_val, cat_features=cat_idx)

        params = cfg['model']['params']
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool, verbose=False)

        proba = model.predict_proba(X_val)[:,1]
        oof_proba[val_idx] = proba
        models.append(model)

        # quick fold metric at 0.5 just for reference
        pred = (proba >= 0.5).astype(int)
        f1 = f1_score(y_val, pred, average='macro')
        print(f"[fold {fold}] macro F1@0.50 = {f1:.4f}")

    # save artifacts
    ensure_dir(paths['model_dir'])
    for i, m in enumerate(models):
        m.save_model(f"{paths['model_dir']}/catboost_fold{i+1}.cbm")
    # Save feature stats (optional) and oof probabilities
    joblib.dump(feat.fit_stats, f"{paths['model_dir']}/feature_stats.joblib")
    pd.DataFrame({'oof_proba': oof_proba, 'y': y}).to_csv(paths['oof_proba_csv'], index=False)
    print("Training complete. Models & OOF saved.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(args.config)
