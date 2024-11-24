import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from datetime import datetime

def initialize_seed(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

def run():
    initialize_seed(52)

    dataset = pd.read_csv('data_daily.csv')
    dataset['Year'] = pd.to_datetime(dataset['# Date']).dt.year
    dataset['Month'] = pd.to_datetime(dataset['# Date']).dt.month
    dataset['Date'] = pd.to_datetime(dataset['# Date'])
    dataset['Day'] = pd.to_datetime(dataset['# Date']).dt.day
    baseline_date = datetime(2021, 1, 1)
    dataset['DeltaDays'] = (dataset['Date'] - baseline_date).dt.days

    past_december_data = dataset[(dataset['Year'] == 2021) & (dataset['Month'] == 12)]
    projected_december_data = past_december_data.copy()
    projected_december_data['Year'] = 2022
    projected_december_data['Date'] = projected_december_data['Date'] + pd.DateOffset(years=1)
    projected_december_data['ReceiptSum'] = (projected_december_data['Receipt_Count'] * 1.3).astype(int)
    projected_december_data['DeltaDays'] = (projected_december_data['DeltaDays'] + 365)

    dataset = pd.concat([dataset, projected_december_data], ignore_index=True)

    features = dataset[['DeltaDays']].values
    targets = dataset['Receipt_Count'].values
    features_train, _, targets_train, _ = train_test_split(features, targets, test_size=0.25, random_state=42)
    features_train = torch.tensor(features_train, dtype=torch.float32)
    targets_train = torch.tensor(targets_train, dtype=torch.float32)

    neural_net = torch.nn.Sequential(
        torch.nn.Linear(1, 64),
        torch.nn.LeakyReLU(negative_slope=0.01),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 1)
    )

    loss_function = torch.nn.MSELoss(reduction='sum')
    optimizer_function = optim.RMSprop(neural_net.parameters(), lr=1e-2)

    num_epochs = 1000
    neural_net.train()
    for epoch_index in range(num_epochs):
        predicted = neural_net(features_train)
        error = loss_function(predicted, targets_train)
        optimizer_function.zero_grad()
        error.backward()
        optimizer_function.step()

    torch.save(neural_net, 'reciept.pth')

if __name__ == "__main__":
    run()