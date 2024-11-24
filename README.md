# **fetch-oa**

A Flask application to host and visualize a machine learning model for predicting receipt counts using Chart.js. This project demonstrates a machine learning pipeline for training, deploying, and visualizing predictions of a neural network model.

---

## **Overview**
The project consists of:
1. **Machine Learning Model**:
   - A neural network trained to predict receipt counts based on historical data.
   - Incorporates data augmentation, normalization, and custom architectural choices for stable and accurate predictions.
2. **Flask Web Application**:
   - A web interface to visualize the model's predictions alongside actual data.
   - Built using Chart.js for interactive and visually appealing graphs.

---

## **Model Details**
### **Architecture**
- **Input Layer**: 1 feature (normalized day count from a baseline date).
- **Hidden Layers**:
  - 64 neurons with LeakyReLU activation.
  - 32 neurons with ReLU activation.
  - 16 neurons with ReLU activation.
  - 4 neurons with ReLU activation.
- **Output Layer**: 1 neuron (predicted receipt count).
  
### **Special Features**
1. **Data Augmentation**:
   - Projected December 2022 data added, assuming a 30% growth rate from December 2021. This provides additional information for the model to learn future trends.
2. **Normalization**:
   - Input (`TimeCnt`) and target (`Receipt_Count`) data are normalized to ensure stable and efficient training.
   - Normalization parameters are saved and used during inference for consistency.
3. **Loss Function**:
   - Mean Squared Error (MSE) with a summation reduction is used to minimize prediction errors.
4. **Optimizer**:
   - RMSprop with a learning rate of 0.01 is used for stable gradient descent updates.

---

## **Code Implementation**
### **1. Model Training**
The `model.py` script:
- Reads and processes historical receipt data.
- Augments data with projections for December 2022.
- Normalizes the input and target values.
- Defines and trains a neural network using PyTorch.
- Saves the trained model and normalization parameters for later use.

To run the model training:

python model.py


### **2. Flask Web Application**

The `predict.py` script:
- Loads the trained model and normalization parameters.
- Processes and normalizes input data.
- Computes predictions and denormalizes the results.
- Renders a webpage displaying predictions and actual receipt counts using Chart.js.

To host the Flask app:

python predict.py

### **3. Web Visualization**

The `index.html` file:
- Fetches processed data from the Flask backend.
- Displays a line graph comparing actual and predicted receipt counts.
- Uses Chart.js for an interactive and responsive visualization.

---

### **Reasoning Behind Design Choices**

1. **Neural Network Architecture**:
   - Chosen for its flexibility in handling non-linear relationships in data.
   - Hidden layers and activations were designed to strike a balance between model complexity and overfitting.

2. **Normalization**:
   - Prevents numerical instability during training.
   - Speeds up convergence by ensuring all features are on a similar scale.

3. **Data Augmentation**:
   - Adding projected data helps the model generalize better for future predictions.

4. **RMSprop Optimizer**:
   - Well-suited for handling noisy data and dynamic learning rates during training.

---

### **Deployment**

#### **Run Locally**

1. **Install Dependencies**:  
   Ensure you have the required Python libraries installed:

   pip install pandas numpy torch scikit-learn flask

2. **Train the Model**:  
   Run the `model.py` script:  

   python model.py

3. **Host the Application**:  
   Launch the Flask web app:

   python predict.py

4. **Access the Web Interface**:
    Open a browser and navigate to `http://localhost:5000`.