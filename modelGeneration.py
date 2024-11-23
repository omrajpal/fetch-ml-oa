import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from datetime import datetime
    

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Set a seed for reproducibility
    set_seed(52) 

    # Loading dataset and extracting relevant variables
    data = pd.read_csv('data_daily.csv')
    data['Year'] = pd.to_datetime(data['# Date']).dt.year
    data['Month'] = pd.to_datetime(data['# Date']).dt.month
    data['Date'] = pd.to_datetime(data['# Date'])
    data['Day'] = pd.to_datetime(data['# Date']).dt.day
    start_date = datetime(2021, 1, 1)
    data['TimeCnt'] = (data['Date'] - start_date).dt.days

    # Adding 'fake' data for December 2022, assuming a 30% growth rate from December 2021 (which is the growth rate from Jan 2021 to Dec 2021)
    december_2021_data = data[(data['Year'] == 2021) & (data['Month'] == 12)]
    december_2022_data = december_2021_data.copy()
    december_2022_data['Year'] = 2022  
    december_2022_data['Date'] = december_2022_data['Date'] + pd.DateOffset(years=1)  
    december_2022_data['Receipt_Count'] = (december_2022_data['Receipt_Count'] * 1.3).astype(int) 
    december_2022_data['TimeCnt'] = (december_2022_data['TimeCnt'] + 365)

    # Concatenate both datasets
    data = pd.concat([data, december_2022_data], ignore_index=True)

    # Remove 25% of data values to prevent overfitting
    X = data[['TimeCnt']].values
    y = data['Receipt_Count'].values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # Create model with 4 linear layers and 3 ReLu activation layers
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 64),
        torch.nn.LeakyReLU(negative_slope = .01),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 1)
    )

    # Use standard loss function for linear layers
    loss_fn = torch.nn.MSELoss(reduction = 'sum')

    # Create optimizer with a learning rate of 1e-2
    optimizer = optim.RMSprop(model.parameters(), lr = 1e-2)


    # Train model
    epochs = 1000
    model.train()
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = loss_fn(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save trained model
    torch.save(model, 'receiptPrediction.pth')

if __name__ == "__main__":
    main()