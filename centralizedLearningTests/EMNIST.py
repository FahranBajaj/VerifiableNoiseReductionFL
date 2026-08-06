import os
import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

#load data
batch_size = 128
transforms = Compose([ToTensor(), Normalize((0.1736,), (0.3317,))])
train_dataset = EMNIST(root = 'data/', download = False, split = "byclass", train = True, transform = transforms)
test_dataset = EMNIST(root = 'data/', download = False, split = "byclass", train = False, transform = transforms)
train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size, shuffle = False)

#define model
class EMNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 5, 100)
        self.fc2 = nn.Linear(100, 62)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 7 * 7 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = EMNIST_CNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.03)
model.train()

FILE_TO_WRITE = "centralizedLearningTests/EMINST.csv"
fieldnames = ["update-round","test-loss","test-accuracy"]
with open(FILE_TO_WRITE, 'a', newline = '') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
    writer.writeheader()

NUM_EPOCHS = 100
for epoch in tqdm(range(NUM_EPOCHS)):
    #Train
    for batch in train_loader:
        optimizer.zero_grad()
        criterion(model(batch[0].to(device)), batch[1].to(device)).backward()
        optimizer.step()

    #Evaluate
    test_correct = 0
    test_loss = 0.0
    for batch in test_loader:
        outputs = model(batch[0].to(device))
        labels = batch[1].to(device)
        test_loss += criterion(outputs, labels).item()
        test_correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    test_accuracy = test_correct / len(test_loader.dataset)
    test_loss = test_loss / len(test_loader)

    results_dict = {
        "update-round": epoch,
        "test-loss": test_loss,
        "test-accuracy": test_accuracy
    }
    with open(FILE_TO_WRITE, 'a', newline = '') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
        writer.writerow(results_dict)    