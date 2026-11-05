import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.models.mlp import MLP
from src.models.gcn import GCN
from src.data_load.load import load_split
from src.evaluate import evaluate_ap, evaluate_auc, summarize_results

torch.set_num_threads(1)

ROOT_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "experiments" / "baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
MODES = ["RL", "RH", "TH"]

CONFIG = {
    "lr": 1e-2,
    "weight_decay": 5e-4,
    "epochs": 300,
    "patience": 50,
    "hidden": 64,
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def inspect_split(data, mode):
    masks = {
        "train": data.train_mask.numpy(),
        "val": data.val_mask.numpy(),
        "test": data.test_mask.numpy(),
    }
    y = data.y.numpy().ravel()
    stats = {"mode": mode, "num_nodes": data.num_nodes,
             "num_edges": data.num_edges, "num_features": data.num_features}
    for name, mask in masks.items():
        stats[f"{name}_size"] = int(mask.sum())
        stats[f"{name}_pos_rate"] = float(y[mask].mean())

    print(f"\n[{mode}] nodes={stats['num_nodes']}, edges={stats['num_edges']}, "
          f"features={stats['num_features']}")
    print(f"[{mode}] train={stats['train_size']} (pos={stats['train_pos_rate']:.3f}), "
          f"val={stats['val_size']} (pos={stats['val_pos_rate']:.3f}), "
          f"test={stats['test_size']} (pos={stats['test_pos_rate']:.3f})")
    return stats


def prepare_features(data, device):
    train_idx = data.train_mask.numpy()
    assert train_idx.any(), "train_mask пуст — проверьте загрузку сплита"
    scaler = StandardScaler().fit(data.x.numpy()[train_idx])
    return torch.tensor(scaler.transform(data.x.numpy()),
                        dtype=torch.float32, device=device)


def train_model(model, forward_fn, data, seed, device):
    set_seed(seed)
    x = prepare_features(data, device)
    edge_index = data.edge_index.to(device)
    y = data.y.float().to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"],
                                 weight_decay=CONFIG["weight_decay"])
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_ap, best_epoch, best_state = -1.0, 0, None
    for epoch in range(CONFIG["epochs"]):
        model.train()
        optimizer.zero_grad()
        criterion(forward_fn(model, x, edge_index)[train_mask],
                  y[train_mask]).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = forward_fn(model, x, edge_index)
            val_ap = evaluate_ap(y[val_mask].cpu().numpy(),
                                 torch.sigmoid(out[val_mask]).cpu().numpy())
        if val_ap > best_val_ap + 1e-6:
            best_val_ap, best_epoch = val_ap, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch > CONFIG["patience"]:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = forward_fn(model, x, edge_index)
        val_pred = torch.sigmoid(out[val_mask]).cpu().numpy()
        test_pred = torch.sigmoid(out[test_mask]).cpu().numpy()

    return {
        "val_AP": evaluate_ap(y[val_mask].cpu().numpy(), val_pred),
        "val_AUC": evaluate_auc(y[val_mask].cpu().numpy(), val_pred),
        "test_AP": evaluate_ap(y[test_mask].cpu().numpy(), test_pred),
        "test_AUC": evaluate_auc(y[test_mask].cpu().numpy(), test_pred),
        "best_epoch": int(best_epoch),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, PyTorch: {torch.__version__}, Seeds: {SEEDS}")

    results = []
    for mode in MODES:
        print(f"\n{'=' * 70}\nРежим: {mode}\n{'=' * 70}")
        data = load_split(split=mode, root=str(ROOT_DIR))
        inspect_split(data, mode)

        for seed in SEEDS:
            print(f"\n--- seed={seed} ---")
            for name, model_cls, fwd in [
                ("MLP", MLP, lambda mdl, x, ei: mdl(x)),
                ("GCN", GCN, lambda mdl, x, ei: mdl(x, ei)),
            ]:
                t0 = time.time()
                model = model_cls(data.num_features,
                                  hidden=CONFIG["hidden"]).to(device)
                m = train_model(model, fwd, data, seed, device)
                results.append({"mode": mode, "model": name, "seed": seed,
                                **m, "time_sec": time.time() - t0,
                                "config": str(CONFIG)})
                print(f"{name:<4} val_AP={m['val_AP']:.4f}, "
                      f"test_AP={m['test_AP']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "baseline_results_raw.csv", index=False)

    summary = summarize_results(df)
    summary.to_csv(OUT_DIR / "baseline_results_summary.csv", index=False)

    print("\n" + "=" * 70 + "\nСВОДКА\n" + "=" * 70)
    for _, r in summary.iterrows():
        print(f"{r['mode']:>3} | {r['model']:<4} | "
        f"val_AP={r['val_AP_mean']:.4f}±{r['val_AP_std']:.4f} | "
        f"test_AP={r['test_AP_mean']:.4f}±{r['test_AP_std']:.4f} | "
        f"val_AUC={r['val_AUC_mean']:.4f} | "
        f"test_AUC={r['test_AUC_mean']:.4f} | "
        f"{r['time_mean']:.1f}s")


if __name__ == "__main__":
    main()