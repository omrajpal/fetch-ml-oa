from flask import Flask, render_template, request
import pandas as pd
import torch
import calendar

app = Flask(__name__)

#Load trained model and dataset
model = torch.load('receiptPrediction.pth')
data = pd.read_csv('data_daily.csv')

# Extract Year and Month from dataset
data['Year'] = pd.to_datetime(data['# Date']).dt.year
data['Month'] = pd.to_datetime(data['# Date']).dt.month

@app.route('/', methods=['GET', 'POST'])
def index():
    output_data = []  
    day = 0

    # Iterate through 2021 and 2022
    for yr in range(2021, 2023):
        # Iterate through each month
        for i in range(1, 13):
            # Calculate number of days in the given month to iterate through
            num_days = calendar.monthrange(yr, i)[1]

            # Initialize value for predicted sum for the month
            predicted_value = 0
            for date in range(0, num_days):
                # Get the predicted # receipts for each day and add it to the month sum
                input_data = torch.tensor([day], dtype=torch.float32)
                with torch.no_grad():
                    predicted_value += model(input_data).item() 
                day+=1

            # Get actual sum for the given month if the year is 2021
            filtered_data = data[data['Month'] == i]
            actual_value = int(filtered_data['Receipt_Count'].sum())
            if(yr == 2022):
                actual_value = 0
            
            # Add data to be rendered in index.html
            output_data.append([int(yr), i, actual_value, predicted_value])


    return render_template('index.html', data=output_data) 


if __name__ == '__main__':
    app.run()
