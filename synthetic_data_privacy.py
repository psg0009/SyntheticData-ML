import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing
def preprocess_data(data):
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    encoder = LabelEncoder()
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Impute missing values
    num_cols = data.select_dtypes(include=[np.number]).columns
    data[num_cols] = imputer.fit_transform(data[num_cols])
    
    # Scale features
    data[num_cols] = scaler.fit_transform(data[num_cols])
    
    # Encode categorical variables
    if 'gender' in data.columns:
        encoded = onehot.fit_transform(data[['gender']])
        encoded_df = pd.DataFrame(encoded, columns=onehot.get_feature_names_out(['gender']))
        data = pd.concat([data.reset_index(drop=True), encoded_df], axis=1).drop(columns=['gender'])
    
    # Encode labels if applicable
    if 'label' in data.columns:
        data['label'] = encoder.fit_transform(data['label'])
    
    return data

# Generate Synthetic Data
def generate_synthetic_data(n_samples):
    np.random.seed(42)
    synthetic_data = pd.DataFrame(np.random.rand(n_samples, 5), columns=[f'feature_{i}' for i in range(5)])
    synthetic_data['label'] = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    return synthetic_data

# XGBoost Model
def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("XGBoost Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    return model

# Membership Inference Attack Simulation
def membership_inference_attack(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    attacker = RandomForestClassifier()
    attacker.fit(X_train, y_train)
    attack_accuracy = attacker.score(X_test, y_test)
    print(f"Membership Inference Attack Accuracy: {attack_accuracy:.4f}")

# Visualization
def visualize_data(data, columns):
    for col in columns:
        sns.kdeplot(data[col].dropna(), shade=True)
        plt.title(f"Distribution of {col}")
        plt.show()

# Differential Privacy (Simple Noise Addition)
def differential_privacy(X, epsilon=0.1):
    noise = np.random.normal(0, epsilon, X.shape)
    return X + noise

if __name__ == "__main__":
    # Generate Synthetic Dataset
    data = generate_synthetic_data(10000)
    data = preprocess_data(data)
    
    if 'label' in data.columns:
        X = data.drop(columns=['label'])
        y = data['label']
    else:
        X = data
        y = pd.Series(np.random.choice([0, 1], len(data)))
    
    # Model Training
    xgb_model = train_xgboost(X, y)
    
    # Membership Inference Attack
    membership_inference_attack(X, y)
    
    # Visualization
    visualize_data(data, ['feature_0', 'feature_1'])
    
    # Differential Privacy Example
    X_priv = differential_privacy(X)
    print("Differential Privacy Applied")
