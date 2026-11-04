from pathlib import Path

from torch_geometric.datasets import GraphLandDataset

# data/ лежит в корне проекта, независимо от текущей рабочей директории
DEFAULT_ROOT = str(Path(__file__).resolve().parents[2] / 'data')


def load_split(name: str = 'tolokers-2', split: str = 'RL', root: str = DEFAULT_ROOT):
    dataset = GraphLandDataset(root=root, name=name, split=split)
    return dataset[0]


def load_thi(name: str = 'tolokers-2', root: str = DEFAULT_ROOT):
    dataset = GraphLandDataset(root=root, name=name, split='THI')
    train_snap, val_snap, test_snap = dataset
    return train_snap, val_snap, test_snap