from flask import Flask, render_template, request
import pandas as pd
import torch
import calendar

app = Flask(__name__)

trained_model = torch.load('receiptPrediction.pth')
dataset = pd.read_csv('data_daily.csv')

dataset['Year_Column'] = pd.to_datetime(dataset['# Date']).dt.year
dataset['Month_Column'] = pd.to_datetime(dataset['# Date']).dt.month

@app.route('/', methods=['GET', 'POST'])
def render_page():
    processed_data = []
    counter = 0

    for year in range(2021, 2023):
        for month in range(1, 13):
            days_in_month = calendar.monthrange(year, month)[1]
            monthly_prediction = 0

            for _ in range(days_in_month):
                normalized_input = torch.tensor([counter], dtype=torch.float32)
                with torch.no_grad():
                    monthly_prediction += trained_model(normalized_input).item()
                counter += 1

            filtered_dataset = dataset[dataset['Month_Column'] == month]
            actual_sum = int(filtered_dataset['Receipt_Count'].sum())
            if year == 2022:
                actual_sum = 0
            
            processed_data.append([int(year), month, actual_sum, monthly_prediction])

    return render_template('index.html', data=processed_data)

if __name__ == '__main__':
    app.run()