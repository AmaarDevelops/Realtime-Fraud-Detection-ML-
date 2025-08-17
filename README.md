# 💳 Credit Card Fraud Detection (Real-Time)

A machine learning project that detects fraudulent transactions in **real-time**, with a clean **frontend interface** for better usability.  
This project applies **data science + machine learning + frontend integration** to simulate how fraud detection systems work in real-world banking applications.

---

## 🚀 Features
- ✅ **Machine Learning Model** trained on credit card transaction dataset  
- ✅ **Real-Time Prediction** capability using backend APIs  
- ✅ **User-Friendly Frontend** (input transaction details, get instant fraud prediction)  
- ✅ **Well-structured pipeline**: preprocessing → training → evaluation → prediction  
- ✅ **Scalable architecture** (can be deployed to the cloud in the future)  

---

## 🛠️ Tech Stack
- **Python (pandas, scikit-learn, numpy, matplotlib, seaborn)** – data handling & ML  
- **Flask / FastAPI** – backend for serving ML model  
- **JavaScript + HTML + CSS** – frontend for real-time input  
- **Joblib / Pickle** – model persistence  

---

## 📊 Dataset
- The project uses the **Kaggle Credit Card Fraud Detection dataset**.  
- Contains anonymized transactions made by European cardholders in September 2013.  
- Highly **imbalanced dataset** (fraud cases ≈ 0.17%).  
- [Dataset Link](https://www.kaggle.com/mlg-ulb/creditcardfraud)  

---

## ⚙️ How It Works
1. **Data Preprocessing** – scaling, splitting, handling imbalance (SMOTE / undersampling)  
2. **Model Training** – tested multiple algorithms (Logistic Regression, Random Forest, XGBoost)  
3. **Evaluation** – accuracy, precision, recall, F1-score, ROC-AUC  
4. **Frontend Integration** – enter transaction details & receive fraud/no-fraud prediction in real time  
5. **Future Scope** – deployment on cloud (Heroku / Render / AWS)  
 

---

## 🎨 Frontend Preview
<img width="1899" height="888" alt="image" src="https://github.com/user-attachments/assets/50eb18c6-cfaa-4a51-bbc5-24306871bc9c" />
<img width="1796" height="879" alt="image" src="https://github.com/user-attachments/assets/bf3dc009-0e32-4640-babe-be5420457213" />
<img width="1920" height="885" alt="image" src="https://github.com/user-attachments/assets/b43be407-c6da-4539-9b57-66e8992301cb" />

** After Entering the information of a Fraud Transaction **
<img width="1724" height="770" alt="image" src="https://github.com/user-attachments/assets/29013483-8a97-4bfe-97b4-2a1f2ba9016e" />








---

## 🚀 Getting Started
### Clone Repository

git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Install Requirements

pip install -r requirements.txt
Run Backend

python app.py
Open Frontend
Navigate to index.html in your browser

Enter transaction details → Get fraud prediction

📌 Future Improvements

🌐 Deploy on Heroku / Render / AWS

📊 Add model explainability (SHAP / LIME)

📱 Create a mobile-friendly UI

⚡ Enhance backend with FastAPI + WebSockets for even faster predictions

🤝 Contributing
Pull requests are welcome! Feel free to fork and improve this project.

🧑‍💻 Author
Amaar

LinkedIn: https://www.linkedin.com/in/amaar-ali-127800343/



⭐ If you like this project, give it a star on GitHub!
