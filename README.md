💳 Real-Time Credit Card Fraud Detection with Machine Learning

Fraudulent transactions are rare, but their impact on financial institutions and customers is massive. This project tackles the credit card fraud detection problem using machine learning, imbalanced learning techniques, and real-time prediction capabilities.

It demonstrates how to:

Build a robust ML pipeline,

Handle extreme class imbalance (fraud <1%),

Deploy fraud detection logic with a Flask-powered frontend for interactive use.

🚀 Key Features

Data Preprocessing

Standard scaling for numerical features.

Robust handling of imbalanced data with SMOTE and ADASYN.

Modeling & Evaluation

Explored Logistic Regression, Random Forest, and XGBoost.

Hyperparameter tuning with GridSearchCV + Stratified K-Fold CV.

Evaluation with imbalanced metrics: ROC-AUC, Precision-Recall, F1-score, Confusion Matrix.

Interpretability

Feature importance visualizations.

Business insights into which features drive fraud detection.

Frontend & Real-Time Prediction

Flask + Jinja2 based web interface.

Clean UI with CSS styling for easy interaction.

Model serialized with joblib for instant fraud predictions.

Deployment-Ready

Dockerized for portability.

Includes Git LFS setup for handling large model artifacts.

📊 Dataset

Source: Kaggle – Credit Card Fraud Detection

Features:

V1 … V28 → PCA-anonymized components

Time, Amount

Target:

Class → 0 = legitimate, 1 = fraud

Challenge: Extremely imbalanced dataset (fraud ~0.17%).

📂 Project Structure
.
├── app.py                       # Flask backend for real-time fraud detection
├── detection.py                 # Training, tuning, and evaluation pipeline
├── train_model.py               # Script for retraining models
├── best_fraud_detection_model.joblib  # Serialized ML model
├── feature_columns.joblib       # Feature schema
├── static/
│   └── style.css                # Styling for the frontend
├── templates/
│   └── index.html               # Jinja2 template for the frontend
├── requirements.txt             # Dependencies
├── Dockerfile                   # For containerized deployment
├── .gitignore
└── .gitattributes

🛠️ Tech Stack

Python 3.x

ML/DS: scikit-learn, imblearn, xgboost, numpy, pandas

Visualization: matplotlib, seaborn

Backend: Flask, Jinja2

Deployment: Docker, Git LFS

Serialization: joblib

⚙️ How to Run Locally

Clone the Repository

git clone https://github.com/AmaarDevelops/Realtime-Fraud-Detection-ML-.git
cd Realtime-Fraud-Detection-ML-


Set Up Virtual Environment

python -m venv venv
source venv/bin/activate   # macOS/Linux
.\venv\Scripts\activate    # Windows


Install Dependencies

pip install -r requirements.txt


Download Dataset from Kaggle and place creditcard.csv in the project root.

Run Training (optional)

python train_model.py


Launch the App

python app.py


Visit http://127.0.0.1:5000/ in your browser.

📈 Results

Best model: XGBoost + ADASYN

Achieved ROC-AUC ≈ 0.985 on test set.

Strong recall on minority class (fraud), reducing false negatives.

Feature importance plots revealed key fraud-driving features.

🌍 Real-World Impact

This system mimics how real fraud detection pipelines work in fintech companies:

Detects fraudulent transactions in near real-time.

Demonstrates how ML can protect both banks and customers from financial loss.

🤝 Contributing

Contributions are welcome!

Fork this repo

Open issues for suggestions or improvements

Submit PRs for enhancements

📧 Contact

For questions, feedback, or collaboration opportunities, connect with me on GitHub: @AmaarDevelops or LinkedIn :-  https://www.linkedin.com/in/amaar-ali-127800343/ .
