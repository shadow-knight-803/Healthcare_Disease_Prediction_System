# 🏥 Prediction of Disease Outbreaks

A **machine learning-powered** web application for predicting **Diabetes, Heart Disease, and Parkinson's Disease** using **Streamlit**.

## 🚀 Project Overview
This project is a **user-friendly web application** that allows users to input **health parameters** and get a **disease prediction**.  
It is built using **Machine Learning (ML) models** trained on real medical datasets.

## 📌 Features
✅ Predicts **Diabetes**, **Heart Disease**, and **Parkinson’s Disease**  
✅ Uses **Logistic Regression**, **Support Vector Machine (SVM)**, and **Random Forest** models  
✅ **Data Preprocessing**: Standardization, Feature Selection, SMOTE for imbalance handling  
✅ **Interactive Web Interface** using **Streamlit**  
✅ **Deployment Ready!**

---

## 🛠️ Technologies Used
- **Python**  
- **Streamlit** (for web app)  
- **Scikit-Learn** (for ML models)  
- **Pandas & NumPy** (for data processing)  
- **Matplotlib & Seaborn** (for visualization)  
- **Pickle** (for saving models)  

---

## 📂 Project Structure
```
📁 Disease_Prediction_WebApp
📂 saved_models/          # Trained machine learning models
📂 datasets/              # CSV datasets for training
📂 images/                # Screenshots of the app
📄 web.py                 # Streamlit web app script
📄 train_diabetes.py       # Model training script (Diabetes)
📄 train_heart.py          # Model training script (Heart Disease)
📄 train_parkinsons.py     # Model training script (Parkinson’s)
📄 requirements.txt        # Required Python libraries
📄 README.md               # Project Documentation (this file)
📄 scaler.sav              # StandardScaler for data preprocessing
```
---

🔧 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/shadow-knight-803/Prediction-of-Disease-Outbreaks

2️⃣ Create & Activate a Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3️⃣ Install Required Packages

pip install -r requirements.txt

4️⃣ Run the Web App

streamlit run web.py

---

## 📊 Model Training & Evaluation
Each disease is predicted using a **different machine learning model**:  
| Disease          | Algorithm Used                   | Accuracy | Precision | Recall | F1-Score |
|-----------------|---------------------------------|----------|-----------|--------|----------|
| **Diabetes**    | Logistic Regression             | 74.03%   | 72.22%    | 66.10% | 74.00%   |
| **Heart Disease** | Support Vector Machine (SVM)   | 86.89%   | 87.50%    | 87.50% | 87.50%   |
| **Parkinson’s**  | Random Forest                  | 94.92%   | 95.56%    | 97.73% | 96.63%   |

## 📌 Evaluation Metrics Explained
- **Accuracy**: Measures the overall correctness of the model.
- **Precision**: Indicates the proportion of true positive cases out of all predicted positives.
- **Recall**: Measures how well the model identifies all actual positive cases.
- **F1-Score**: Harmonic mean of precision and recall, balancing the two metrics.

## 🔍 Detailed Classification Reports
### **Diabetes Model (Logistic Regression)**
```
               precision    recall  f1-score   support

           0       0.83      0.75      0.79       100
           1       0.61      0.72      0.66        54

    accuracy                           0.74       154
   macro avg       0.72      0.74      0.73       154
weighted avg       0.75      0.74      0.74       154
```

### **Heart Disease Model (SVM)**
```
               precision    recall  f1-score   support

           0       0.87      0.86      0.87        29
           1       0.88      0.88      0.88        32

    accuracy                           0.87        61
   macro avg       0.87      0.87      0.87        61
weighted avg       0.87      0.87      0.87        61

Confusion Matrix:
 [[25  4]
  [ 4 28]]
```

### **Parkinson’s Model (Random Forest)**
```
Accuracy: 0.9492
Precision: 0.9556
Recall: 0.9773
F1-score: 0.9663

Classification Report:
               precision    recall  f1-score   support

           0       0.93      0.87      0.90        15
           1       0.96      0.98      0.97        44

    accuracy                           0.95        59
   macro avg       0.94      0.92      0.93        59
weighted avg       0.95      0.95      0.95        59

Confusion Matrix:
 [[13  2]
  [ 1 43]]
```

Cross-validated Accuracy: 81.54%

### 🔧 Future Enhancements
- Improve **feature selection** and data preprocessing.
- Experiment with **deep learning models** for better predictions.
- Optimize **hyperparameters** further to enhance performance.

---

## 🌟 Future Improvements
- **Improve Model Accuracy** with advanced feature selection  
- **Deploy Online** using Streamlit Cloud or AWS  
- **Add More Diseases** like cancer or lung disease  

---

## 💡 Contributing
🙌 Contributions are welcome!  
- Fork the repository  
- Create a new branch  
- Submit a pull request  

---

## 📜 License
This project is licensed under the **MIT License**. Feel free to use and modify it.

---

## 📞 Contact
🔗 **GitHub**: https://github.com/shadow-knight-803
📧 **Email**: adityaarya803@gmail.com  

