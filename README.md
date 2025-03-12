# SyntheticData-ML

This repository contains the implementation of the research paper **"Evaluating the Utility of Synthetic Patient Data in Machine Learning-Based Disease Prediction: A Security, Fairness, and Privacy Perspective"**. The project evaluates the use of **synthetic data, differential privacy, federated learning, and adversarial robustness** in healthcare AI models.

## Overview
This repository includes:
- **Data Preprocessing**: Handling missing values, normalization, and encoding.
- **Synthetic Data Generation**: Using **CTGAN** and **MedGAN**.
- **Model Training**:
  - **XGBoost** for disease prediction.
  - **LSTM** for sequential EHR data.
- **Privacy and Security Analysis**:
  - **Membership Inference Attacks**.
  - **Differential Privacy Implementation**.
  - **Adversarial Attacks (FGSM)**.

## Repository Structure
```
├── data/                     # Dataset files (synthetic & real EHRs)
├── models/                   # Trained models & saved checkpoints
├── src/                      # Source code
│   ├── synthetic_data_privacy.py  # Main implementation script
├── results/                  # Experimental results & plots
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies
└── LICENSE                   # License file
```

## Getting Started
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Privacy-Security-MedAI.git
cd Privacy-Security-MedAI
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Main Implementation
```bash
python src/synthetic_data_privacy.py
```

##  Results & Analysis
Key findings from our experiments:
- **Membership Inference Attack Accuracy**: ~66.5%
- **Differential Privacy Effect**: Reduces re-identification risk while maintaining **~95% model accuracy**.
- **Adversarial Attack Effect**: FGSM reduces model accuracy by **10–15%**, exposing vulnerabilities.

## Visualizations
**Figure 1:** Membership Inference Attack Accuracy and Model Vulnerability.
![Membership Inference Attack]![image](https://github.com/user-attachments/assets/ec0443e5-97d7-455b-9848-09e020692b9a)



**Figure 2:** Data Distribution Before and After Differential Privacy Application.
![Differential Privacy Applied]![image](https://github.com/user-attachments/assets/f80596a1-fd48-4baa-a751-2f6381b518e5)



## Privacy-Preserving AI Techniques
This project integrates several security enhancements:
- **Synthetic Data Usage**: Reduces the need for real patient records.
- **Federated Learning**: Model training across multiple institutions without sharing data.
- **Differential Privacy**: Controlled noise addition prevents patient re-identification.
- **Adversarial Robustness**: Protecting AI models from attacks.
