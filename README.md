# Neural Network Churn Modelling

## Overview
This project implements a neural network model to predict customer churn using the Churn_Modelling.csv dataset. The model uses TensorFlow/Keras to build an artificial neural network (ANN) for binary classification.

## Dataset
- **Source**: Churn_Modelling.csv
- **Target Variable**: Customer churn (Exited column)
- **Features**: 11 independent variables including customer demographics, account details, and behavioral data

## Data Preprocessing

### Feature Selection
- Selected columns 3-12 as independent features (X)
- Selected column 13 (Exited) as dependent variable (y)
- Features include: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

### Feature Engineering
- **One-Hot Encoding**: Converted categorical variables to dummy variables
  - Geography: Created Germany and Spain columns (dropped France to avoid multicollinearity)
  - Gender: Created Male column (dropped Female)
- **Data Concatenation**: Combined original dataset with dummy variables

### Data Splitting
- **Training Set**: 80% of data (8,000 samples)
- **Test Set**: 20% of data (2,000 samples)
- Used `train_test_split` with `random_state=0` for reproducibility

### Feature Scaling
- Applied StandardScaler to normalize numerical features
- **Why Scaling**: Essential for neural networks as they are distance-based algorithms
- Scaled features: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Germany, Spain, Male

## Neural Network Architecture

### Model Structure

- Input Layer: 11 neurons (matching number of features)
- Hidden Layer 1: 7 neurons with ReLU activation + Dropout(0.2)
- Hidden Layer 2: 6 neurons with ReLU activation
- Output Layer: 1 neuron with Sigmoid activation (binary classification)


### Model Configuration
- **Optimizer**: Adam with learning rate 0.01
- **Loss Function**: Binary crossentropy
- **Metrics**: Accuracy
- **Activation Functions**: 
  - ReLU for hidden layers
  - Sigmoid for output layer

### Training Configuration
- **Batch Size**: 10
- **Epochs**: 1000 (with early stopping)
- **Validation Split**: 33%
- **Early Stopping**: 
  - Monitor: validation loss
  - Patience: 20 epochs
  - Min delta: 0.001

## Model Performance

### Training Results
- The model achieved good performance with early stopping preventing overfitting
- Training showed consistent improvement in both accuracy and loss

### Test Results
- **Confusion Matrix**:
  ```
  [[1506   89]
   [ 195  210]]
  ```
- **Accuracy**: 85.8%
- **True Negatives**: 1506
- **False Positives**: 89
- **False Negatives**: 195
- **True Positives**: 210

## Key Insights

### Model Strengths
- High accuracy (85.8%) on test data
- Good balance between precision and recall
- Effective use of early stopping to prevent overfitting
- Proper feature scaling and encoding

### Technical Implementation
- Used TensorFlow 2.20.0
- Implemented proper train/validation/test split
- Applied feature scaling for neural network optimization
- Used dropout for regularization
- Implemented early stopping for better generalization

## Libraries Used
- **TensorFlow/Keras**: Neural network implementation
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Scikit-learn**: Data preprocessing and model evaluation
- **Matplotlib**: Visualization

## Conclusion
The neural network model successfully predicts customer churn with 85.8% accuracy. The implementation demonstrates proper machine learning practices including data preprocessing, feature engineering, model architecture design, and performance evaluation.
