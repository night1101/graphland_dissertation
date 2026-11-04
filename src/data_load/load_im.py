from torch_geometric.datasets import GraphLandDataset

ROOT_DIR = '../../data' 

dataset_rl = GraphLandDataset(root=ROOT_DIR, name='tolokers-2', split='RL')
data_rl = dataset_rl[0]

print("\nОбъект Data для RL:")
print(data_rl)

dataset_rh = GraphLandDataset(root=ROOT_DIR, name='tolokers-2', split='RH')
data_rh = dataset_rh[0]
print("\nОбъект Data для RH:")
print(data_rh)

dataset_th = GraphLandDataset(root=ROOT_DIR, name='tolokers-2', split='TH')
data_th = dataset_th[0]
print("\nОбъект Data для TH:")
print(data_th)


dataset_thi = GraphLandDataset(root=ROOT_DIR, name='tolokers-2', split='THI')
train_snapshot, val_snapshot, test_snapshot = dataset_thi

print("\nTHI Train snapshot:")
print(train_snapshot)
print("\nTHI Val snapshot:")
print(val_snapshot)
print("\nTHI Test snapshot:")
print(test_snapshot)
