# **SMS Spam Detection System**

## **Live Application**
**https://spamsmsdetectionbynaman.streamlit.app/**

---

## **Project Overview**
This project is an **end-to-end SMS Spam Detection system** that classifies text messages as **Spam** or **Not Spam (Ham)** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
The system is trained on real SMS data, evaluated using multiple classifiers, and deployed as an interactive web application.

---

## **Key Features**
- **End-to-end NLP pipeline** from raw text to prediction  
- **Robust text preprocessing** including tokenization, stopword removal, and stemming  
- **TF-IDF vectorization** for numerical text representation  
- **Multiple machine learning models** trained and evaluated  
- **High-precision spam detection** using Multinomial Naive Bayes  
- **Live deployment** using Streamlit  

---

## **Dataset**
- **SMS Spam Collection Dataset**
- Contains labeled SMS messages as **spam** or **ham**
- Cleaned and deduplicated before training

---

## **Text Preprocessing**
- **Lowercasing**
- **Tokenization**
- **Stopword removal**
- **Stemming**

---

## **Feature Engineering**
- **TF-IDF (Term Frequency–Inverse Document Frequency)**
- Top **3000 most informative features**

---

## **Models Trained**
- **Multinomial Naive Bayes**
- Logistic Regression
- Support Vector Classifier
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

---

## **Model Selection**
**Multinomial Naive Bayes** was selected due to its **high precision**, **fast training**, and excellent performance on sparse TF-IDF features.

---

## **Evaluation Metrics**
- **Accuracy**
- **Precision**
- **Confusion Matrix**

Precision was prioritized to minimize false classification of genuine messages as spam.

---

## **Web Application**
The trained model is deployed as a **Streamlit web application** that allows users to enter an SMS message and instantly receive a **Spam** or **Not Spam** prediction.

---

## **Tech Stack**
- **Python**
- **Pandas**
- **NumPy**
- **NLTK**
- **Scikit-learn**
- **Streamlit**

---

## **Project Structure**
SpamSmsDetection/

├── app.py

├── model.pkl


├── vectorizer.pkl

├── requirements.txt

├── spam.csv

├── notebook.ipynb

└── README.md

---

## **How to Run Locally**
```bash
pip install -r requirements.txt
streamlit run app.py
