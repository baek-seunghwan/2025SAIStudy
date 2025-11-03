from __future__ import annotations
import argparse, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from .features import FeatureBuilder
from .utils import load_config

def positive_quota_threshold(proba: np.ndarray, quota: int) -> float:
    quota = max(1, min(len(proba)-1, int(quota)))
    thr = np.partition(proba, -quota)[-quota]
    return float(thr)

def max_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    # sweep thresholds on unique probs
    thrs = np.unique(proba)
    best_thr, best_f1 = 0.5, -1.0
    for t in thrs:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, average='macro')
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_thr)

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    paths = cfg['paths']
    df = pd.read_csv(paths['train_csv'])
    y = df['fraud'] if 'fraud' in df.columns else df.iloc[:, -1]
    X = df.drop(columns=['fraud']) if 'fraud' in df.columns else df.iloc[:, :-1]

    feat = FeatureBuilder()
    X = feat.fit_transform(X)

    # train simple CV to collect OOF proba (if not cached)
    oof_path = paths.get('oof_proba_csv', 'submissions/oof_proba.csv')
    if pd.notna(oof_path) and not pd.Path(oof_path).exists():
        pass  # simple path check, but we'll just regenerate below

    skf = StratifiedKFold(n_splits=cfg['cv']['n_splits'], shuffle=cfg['cv']['shuffle'], random_state=cfg['seed'])
    oof_proba = np.zeros(len(X), dtype=float)

    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    for trn_idx, val_idx in skf.split(X, y):
        train_pool = Pool(X.iloc[trn_idx], y.iloc[trn_idx], cat_features=cat_idx)
        valid_pool = Pool(X.iloc[val_idx], y.iloc[val_idx], cat_features=cat_idx)
        model = CatBoostClassifier(**cfg['model']['params'])
        model.fit(train_pool, eval_set=valid_pool, verbose=False)
        oof_proba[val_idx] = model.predict_proba(valid_pool)[:,1]

    # strategies
    strategy = cfg['threshold']['strategy']
    if strategy == "max_f1":
        best_thr = max_f1_threshold(y.values, oof_proba)
        grid_info = [{"thr": best_thr, "macro_f1": f1_score(y, (oof_proba>=best_thr).astype(int), average='macro')}]
    else:
        grid = cfg['threshold']['positive_quota_grid']
        grid_info = []
        for q in grid:
            thr = positive_quota_threshold(oof_proba, quota=q)
            f1 = f1_score(y, (oof_proba>=thr).astype(int), average='macro')
            grid_info.append({"target_pos": q, "thr": thr, "macro_f1": f1})

        # choose by best f1 on OOF
        best = max(grid_info, key=lambda d: d["macro_f1"])
        best_thr = best["thr"]

    # write grid results
    res_df = pd.DataFrame(grid_info)
    res_df.to_csv("submissions/threshold_search_results.csv", index=False)
    print("Saved grid to submissions/threshold_search_results.csv")
    print("Selected thr:", best_thr)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(args.config)
