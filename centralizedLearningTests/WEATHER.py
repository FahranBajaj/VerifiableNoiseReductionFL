import os
import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

weather_data = pd.read_csv("data/weatherAUS.csv")
weather_data["Year"] = weather_data["Date"].apply(lambda ymd: int(ymd.split("-")[0]))
weather_data["RainToday"] = weather_data["RainToday"].apply(lambda x: 1 if x == "Yes" else 0)
weather_data["RainTomorrow"] = weather_data["RainTomorrow"].apply(lambda x: 1 if x == "Yes" else 0)
cols_to_use = [
    "Date", 
    "Location", 
    "Year", 
    "MinTemp", 
    "MaxTemp", 
    "Rainfall", 
    "WindSpeed9am", 
    "WindSpeed3pm", 
    "Humidity9am", 
    "Humidity3pm", 
    "Pressure9am", 
    "Pressure3pm", 
    "Temp9am", 
    "Temp3pm", 
    "RainToday", 
    "RainTomorrow"
]
weather_data = weather_data[cols_to_use].dropna()
scaler = StandardScaler()
weather_data[cols_to_use[3:14]] = scaler.fit_transform(weather_data[cols_to_use[3:14]])
train_data: pd.DataFrame = weather_data[weather_data["Year"] < 2016][cols_to_use[3:]]
test_data: pd.DataFrame = weather_data[weather_data["Year"] >= 2016][cols_to_use[3:]]

class WeatherDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.Tensor(np.array(data))
        self.targets = torch.Tensor(np.array(targets)).type(torch.LongTensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

batch_size = 128
train_dataset = WeatherDataset(train_data.drop(columns = "RainTomorrow"), train_data["RainTomorrow"])
test_dataset = WeatherDataset(test_data.drop(columns = "RainTomorrow"), test_data["RainTomorrow"])
train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size, shuffle = False)

class WeatherModel(nn.Module):
    def __init__(self):
        super(WeatherModel, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = WeatherModel().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.03)
model.train()

FILE_TO_WRITE = "centralizedLearningTests/WEATHER.csv"
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