from pathlib import Path

from torch_geometric.datasets import GraphLandDataset

ROOT_DIR = '../../data' 

def load_split(name: str = 'tolokers-2', split: str = 'RL', root: str = ROOT_DIR):
    dataset = GraphLandDataset(root=root, name=name, split=split)
    return dataset[0]


def load_thi(name: str = 'tolokers-2', root: str = ROOT_DIR):
    dataset = GraphLandDataset(root=root, name=name, split='THI')
    train_snap, val_snap, test_snap = dataset
    return train_snap, val_snap, test_snap
