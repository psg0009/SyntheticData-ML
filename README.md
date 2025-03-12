# SyntheticData-ML

# Privacy-Preserving Medical AI Research

This repository contains the implementation of the research paper **"Enhancing Privacy and Security in Medical AI: A Comprehensive Study on Synthetic Data and Model Robustness"**. The project evaluates the use of **synthetic data, differential privacy, federated learning, and adversarial robustness** in healthcare AI models.

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
![Membership Inference Attack](attachment:file-J97ToLbAqJWcQxYtzrhzqY)

**Figure 2:** Data Distribution Before and After Differential Privacy Application.
![Differential Privacy Applied](attachment:file-HggiwhJaB92bZz2weyHPub)

## Privacy-Preserving AI Techniques
This project integrates several security enhancements:
- **Synthetic Data Usage**: Reduces the need for real patient records.
- **Federated Learning**: Model training across multiple institutions without sharing data.
- **Differential Privacy**: Controlled noise addition prevents patient re-identification.
- **Adversarial Robustness**: Protecting AI models from attacks.
