import json
import os
import subprocess
import sys
import tempfile
import time

# Позволяет запускать как `python src/train.py`, так и `python -m src.train`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.models.mlp import MLP
from src.models.gcn import GCN
from src.data_load.load import load_split
from src.evaluate import evaluate_ap, evaluate_auc, summarize_results

ROOT_DIR = os.path.join(PROJECT_ROOT, 'data')
OUT_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'baseline')
LGB_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lgb_worker.py')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [0, 1, 2]

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_lightgbm(data, seed):
    X = data.x.numpy()
    y = data.y.numpy()
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    # LightGBM обучается в отдельном процессе: на macOS torch и lightgbm
    # загружают разные libomp, и их совместная работа в одном процессе
    # приводит к segfault.
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        np.savez(tmp, X_train=X_train, y_train=y_train,
                 X_val=X_val, y_val=y_val)
        data_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, LGB_WORKER, data_path, '--seed', str(seed)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    finally:
        os.remove(data_path)

    if proc.returncode != 0:
        raise RuntimeError(
            f'LightGBM worker exited with code {proc.returncode}:\n{proc.stderr}'
        )

    result = json.loads(proc.stdout.strip().splitlines()[-1])
    return result['ap'], result['auc'], result['best_iteration']

def run_mlp(data, seed, device='cpu', epochs=300, patience=50):
    set_seed(seed)
    # Стандартизация признаков: fit ТОЛЬКО на train, чтобы не было утечки
    scaler = StandardScaler()
    x_np = data.x.numpy()
    train_mask = data.train_mask.numpy()
    scaler.fit(x_np[train_mask])
    x_scaled = torch.tensor(scaler.transform(x_np), dtype=torch.float32).to(device)

    y = data.y.float().to(device)
    train_mask_t = data.train_mask.to(device)
    val_mask_t = data.val_mask.to(device)

    model = MLP(data.num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_ap, best_epoch, best_state = -1, 0, None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_scaled)
        loss = criterion(out[train_mask_t], y[train_mask_t])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x_scaled)
            val_pred = torch.sigmoid(out[val_mask_t]).cpu().numpy()
            val_true = y[val_mask_t].cpu().numpy()
            val_ap = evaluate_ap(val_true, val_pred)
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch > patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(x_scaled)
        val_pred = torch.sigmoid(out[val_mask_t]).cpu().numpy()
        val_true = y[val_mask_t].cpu().numpy()
    return evaluate_ap(val_true, val_pred), evaluate_auc(val_true, val_pred), best_epoch

def run_gcn(data, seed, device='cpu', epochs=300, patience=50):
    set_seed(seed)
    # Нормализация признаков ТОЛЬКО по train
    scaler = StandardScaler()
    x_np = data.x.numpy()
    train_mask = data.train_mask.numpy()
    scaler.fit(x_np[train_mask])
    x_scaled = torch.tensor(scaler.transform(x_np), dtype=torch.float32).to(device)

    edge_index = data.edge_index.to(device)
    y = data.y.float().to(device)
    train_mask_t = data.train_mask.to(device)
    val_mask_t = data.val_mask.to(device)

    model = GCN(data.num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_ap, best_epoch, best_state = -1, 0, None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_scaled, edge_index)
        loss = criterion(out[train_mask_t], y[train_mask_t])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x_scaled, edge_index)
            val_pred = torch.sigmoid(out[val_mask_t]).cpu().numpy()
            val_true = y[val_mask_t].cpu().numpy()
            val_ap = evaluate_ap(val_true, val_pred)
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch > patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(x_scaled, edge_index)
        val_pred = torch.sigmoid(out[val_mask_t]).cpu().numpy()
        val_true = y[val_mask_t].cpu().numpy()
    return evaluate_ap(val_true, val_pred), evaluate_auc(val_true, val_pred), best_epoch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    results = []
    for mode in ['RL', 'RH', 'TH']:
        print(f"\n{'='*60}\nРежим: {mode}\n{'='*60}")
        data = load_split(split=mode, root=ROOT_DIR)

        for seed in SEEDS:
            print(f"\n--- seed={seed} ---")
            t0 = time.time()

            # LightGBM
            try:
                ap, auc, best_it = run_lightgbm(data, seed)
                results.append({'mode': mode, 'model': 'LightGBM',
                                'seed': seed, 'val_AP': ap, 'val_AUC': auc,
                                'extra': f'best_iter={best_it}',
                                'time_sec': time.time() - t0})
                print(f"LightGBM: AP={ap:.4f}, AUC={auc:.4f}, iter={best_it}")
            except Exception as e:
                print(f"LightGBM FAILED: {e}")

            t0 = time.time()
            ap, auc, best_ep = run_mlp(data, seed, device=device)
            results.append({'mode': mode, 'model': 'MLP',
                            'seed': seed, 'val_AP': ap, 'val_AUC': auc,
                            'extra': f'best_epoch={best_ep}',
                            'time_sec': time.time() - t0})
            print(f"MLP:      AP={ap:.4f}, AUC={auc:.4f}, epoch={best_ep}")

            t0 = time.time()
            ap, auc, best_ep = run_gcn(data, seed, device=device)
            results.append({'mode': mode, 'model': 'GCN',
                            'seed': seed, 'val_AP': ap, 'val_AUC': auc,
                            'extra': f'best_epoch={best_ep}',
                            'time_sec': time.time() - t0})
            print(f"GCN:      AP={ap:.4f}, AUC={auc:.4f}, epoch={best_ep}")

    # ---------- 5. Сводная таблица ----------
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, 'baseline_results_raw.csv'), index=False)

    summary = summarize_results(df)
    summary.to_csv(os.path.join(OUT_DIR, 'baseline_results_summary.csv'), index=False)

    print("\n" + "="*60)
    print("СВОДНАЯ ТАБЛИЦА (mean ± std по seed)")
    print("="*60)
    for _, row in summary.iterrows():
        print(f"{row['mode']:>3} | {row['model']:<10} | "
              f"AP={row['val_AP_mean']:.4f}±{row['val_AP_std']:.4f} | "
              f"AUC={row['val_AUC_mean']:.4f}±{row['val_AUC_std']:.4f} | "
              f"{row['time_mean']:.1f}s")

if __name__ == '__main__':
    main()