import torch
from torch.utils.data import Dataset, DataLoader

data = {"train0": torch.Tensor([
        [1, 1],
        [0.5, 3],
        [-1, 2],
        [-1, -1],
        [0.5, -3],
        [1.5, 0.5]]),
    "test0": torch.Tensor([
        [-2, -0.5],
        [0.5, -2]]),
    "train1": torch.Tensor([
        [-1, 2.5],
        [-1, -2],
        [3, -1.5],
        [3, 1.5],
        [-1.5, 2],
        [-2.5, -1]]),
    "test1": torch.Tensor([
        [1, -3],
        [0.5, 2.5]])
}

targets = {"train0": torch.Tensor([
        [1,0],
        [1,0],
        [0,1],
        [1,0],
        [0,1],
        [1,0]]),
    "test0": torch.Tensor([
        [1,0],
        [0,1]]),
    "train1": torch.Tensor([
        [0,1],
        [1,0],
        [0,1],
        [1,0],
        [0,1],
        [1,0]]),
    "test1": torch.Tensor([
        [0,1],
        [1,0]])
}

class XORDataset(Dataset):
    def __init__(self, central, task, id):
        if central:
            self.data = torch.cat((data[task + "0"], data[task + "1"]))
            self.targets = torch.cat((targets[task + "0"], targets[task + "1"]))
        else:
            self.data = data[task + str(id % 2)]
            self.targets = targets[task + str(id % 2)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    return DataLoader(XORDataset(False, "train", partition_id), batch_size), DataLoader(XORDataset(False, "test", partition_id), batch_size)

def load_centralized_dataset():
    #TODO: find a good way to manage batch size?
    return DataLoader(XORDataset(True, "test", 0))