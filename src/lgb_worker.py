"""Обучение LightGBM в отдельном процессе.

На macOS PyTorch и LightGBM каждый загружают свою копию libomp
(OpenMP). Если обе оказываются в одном процессе, при обучении
происходит segfault. Поэтому LightGBM запускается отдельным
процессом, который не импортирует torch.
"""

import argparse
import json

import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='.npz с X_train, y_train, X_val, y_val')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    with np.load(args.data) as f:
        X_train = f['X_train']
        y_train = f['y_train']
        X_val = f['X_val']
        y_val = f['y_val']

    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='average_precision',
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    val_score = model.predict_proba(X_val)[:, 1]
    result = {
        'ap': float(average_precision_score(y_val, val_score)),
        'auc': float(roc_auc_score(y_val, val_score)),
        'best_iteration': int(model.best_iteration_),
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
