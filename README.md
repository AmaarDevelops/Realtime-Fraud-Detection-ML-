Credit Card Fraud Detection: An Advanced Machine Learning Approach
🌟 Project Overview

This project focuses on developing, evaluating, and optimizing machine learning models for credit card fraud detection. Given the highly imbalanced nature of real-world financial transaction datasets (where fraudulent transactions are a tiny fraction of legitimate ones), a core objective was to implement strategies to effectively identify the rare fraud cases. The project demonstrates a robust machine learning pipeline from data preprocessing and imbalance handling to model training, hyperparameter tuning, and comprehensive evaluation.

🚀 Key Features

Data Preprocessing: Standard scaling of numerical features to ensure optimal model performance.

Imbalanced Learning: Application and evaluation of oversampling techniques (SMOTE, ADASYN) to address the severe class imbalance inherent in fraud detection datasets.

Model Exploration: Implementation and comparison of various classification algorithms, including:

Logistic Regression

XGBoost (Extreme Gradient Boosting)

Random Forest Classifier

Hyperparameter Tuning: Utilization of GridSearchCV with StratifiedKFold cross-validation to find the optimal parameters for the best-performing models, ensuring robust evaluation on imbalanced data.

Comprehensive Evaluation: Assessment of models using critical metrics for imbalanced datasets:

ROC AUC Score

Classification Report (Precision, Recall, F1-score for both classes)

Confusion Matrix

Feature Importance Analysis: Identification of the most influential features contributing to fraud prediction for interpretability.

Production-Ready Assets: Saving of the best-trained model and a list of expected feature columns for potential future deployment as an API.

📊 Dataset
The dataset used in this project is the Credit Card Fraud Detection dataset from Kaggle. It contains anonymized credit card transactions labelled as either legitimate (Class = 0) or fraudulent (Class = 1).

Features: V1, V2, ..., V28 (anonymized principal components), Time, and Amount.

Target Variable: Class (0 for legitimate, 1 for fraud).

Characteristics: Highly imbalanced, with fraudulent transactions accounting for a very small percentage of the total.

📂 Project Structure
.
├── detection.py              # Main script for model training, tuning, and evaluation
├── best_fraud_detection_model.joblib # Saved serialized best-performing ML pipeline
├── feature_columns.joblib      # Saved list of feature columns in correct order
├── static / style .css         # CSS For Styling
├──templates / index.html        #Consists the html file for Frontend
├── app.py                      # Main Flask Backend
├── requirements.txt            # List of Python dependencies
└── Docker File
├── .gitignore
├── .gitattributes

🛠️ Technologies Used
Python 3.x

pandas (for data manipulation)

numpy (for numerical operations)

scikit-learn (for machine learning models, preprocessing, pipelines, and evaluation)

imblearn (for handling imbalanced datasets like SMOTE, ADASYN, and ImbPipeline)

xgboost (for the XGBoost classifier)

matplotlib (for plotting and visualizations)

seaborn (for enhanced statistical data visualization)

joblib (for model serialization)

⚙️ How to Run Locally
To replicate the model training and evaluation process on your local machine, follow these steps:

1. Clone the Repository
git clone https://github.com/AmaarDevelops/Realtime-Fraud-Detection-ML-.git
cd Realtime-Fraud-Detection-ML-

2. Install Git LFS
This project uses Git Large File Storage (LFS) to handle large binary files (your saved model).

Install Git LFS: Follow instructions on Git LFS website. For example, on macOS with Homebrew:

brew install git-lfs

Initialize Git LFS in your repository:

git lfs install

Pull LFS files: Ensure the actual large files are downloaded:

git lfs pull

3. Download the Dataset
Go to the Credit Card Fraud Detection dataset on Kaggle.

Download the creditcard.csv file.

Place creditcard.csv directly into the root directory of this project.

4. Set Up a Python Virtual Environment (Recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

5. Install Dependencies
Navigate to the project root directory (where requirements.txt is located) and install all necessary Python packages:

pip install -r requirements.txt

6. Run the Training Script
Execute the main script to train, tune, and evaluate the models. This process will also save the best model (best_fraud_detection_model.joblib) and feature columns (feature_columns.joblib) in the project root.

python train_model.py

The script will print detailed evaluation reports, confusion matrices, ROC AUC scores, and display feature importance plots in your terminal.

📈 Results and Evaluation

The project thoroughly explores various models and imbalance handling techniques. The evaluation highlights the effectiveness of using techniques like ADASYN and SMOTE with ensemble methods (XGBoost, Random Forest) to significantly improve the detection of fraudulent transactions.


ROC Curves:
The Receiver Operating Characteristic (ROC) curves provide a clear visual comparison of the true positive rate versus the false positive rate for different models.

Feature Importance:
Understanding which features are most indicative of fraud is crucial for interpretability and potentially for feature engineering.

The  XGBoost with ADASYN Model consistently demonstrated superior performance, achieving an ROC AUC of approximately 0.985 on the test set, indicating its strong ability to distinguish between fraudulent and legitimate transactions even in a highly imbalanced environment.

🤝 Contributing
Feel free to fork this repository, explore the code, suggest improvements, or open issues. Contributions are welcome!

📧 Contact
For any questions or collaborations, please reach out via GitHub.
