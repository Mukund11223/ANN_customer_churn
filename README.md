# Customer Churn Prediction using Artificial Neural Network (ANN)

## Description
This project predicts customer churn for a banking institution using an Artificial Neural Network (ANN). The model processes customer data to predict whether a customer is likely to leave the bank (churn). The dataset includes features like credit score, geography, gender, age, tenure, balance, and more.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction

### Install required libraries:

bash
pip install pandas numpy scikit-learn tensorflow

## Usage
### Data Preprocessing:
- Load the dataset Churn_Modelling.csv.
- Drop irrelevant columns (RowNumber, CustomerId, Surname).
- Encode categorical variables (Gender with LabelEncoder, Geography with OneHotEncoder).
- Scale features using StandardScaler.

### Model Training:

Run the script to train the ANN:

bash

python churn_prediction.py

## The model architecture:
- Input layer: 12 features.
- Two hidden layers (64 and 32 neurons, ReLU activation).
- Output layer: 1 neuron with sigmoid activation for binary classification.
 
### Prediction:

Use the saved model (model.h5), scaler (scaler.pkl), and encoders to predict churn on new data:

# Example input
<img width="223" alt="Screenshot 2025-02-03 at 8 21 13 PM" src="https://github.com/user-attachments/assets/75200fcf-d6d8-471f-ae60-3d16ce375291" />

# Preprocess input and predict
prediction = model.predict(preprocessed_input)
Files
churn_prediction.py: Main script for data preprocessing, model training, and saving artifacts.

model.h5: Trained ANN model.

scaler.pkl, le_gender.pkl, One_geo_encoder.pkl: Preprocessing artifacts.

Churn_Modelling.csv: Dataset (not included; download link).

Model Architecture
Copy
Model: "sequential_1"

<img width="555" alt="Screenshot 2025-02-03 at 8 20 32 PM" src="https://github.com/user-attachments/assets/24de79e5-8035-4ddf-a185-756d708d098d" />
  
Total params: 2,945 (11.50 KB)
Training
Optimizer: Adam (learning_rate=0.01).

Loss Function: Binary cross-entropy.

### Callbacks:

- EarlyStopping (patience=10) to prevent overfitting.
- TensorBoard for training visualization.
- Validation Accuracy: ~85-86% (observed during training).

## Results
The model achieves ~85% validation accuracy.



