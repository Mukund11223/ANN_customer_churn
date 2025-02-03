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
-Scale features using StandardScaler.

### Model Training:

Run the script to train the ANN:

bash

python churn_prediction.py

## The model architecture:
- Input layer: 12 features.
-Two hidden layers (64 and 32 neurons, ReLU activation).
-Output layer: 1 neuron with sigmoid activation for binary classification.

### Prediction:

Use the saved model (model.h5), scaler (scaler.pkl), and encoders to predict churn on new data:

# Example input
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}
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
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_3 (Dense)                 │ (None, 64)             │           832 │
│ dense_4 (Dense)                 │ (None, 32)             │         2,080 │
│ dense_5 (Dense)                 │ (None, 1)              │            33 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
Total params: 2,945 (11.50 KB)
Training
Optimizer: Adam (learning_rate=0.01).

Loss Function: Binary cross-entropy.

### Callbacks:

- EarlyStopping (patience=10) to prevent overfitting.
- TensorBoard for training visualization.
- Validation Accuracy: ~85-86% (observed during training).

## Results
The model achieves ~85% validation accuracy. Example prediction:



