# Heart Disease Prediction 🚀  

This project is a complete **Machine Learning pipeline** for predicting the likelihood of heart disease using patient health data. It includes **data preprocessing, feature engineering, dimensionality reduction, model training, evaluation, and deployment via Streamlit.**  

## 🌍 Live Demo  
You can try the deployed app here:  
👉 [Heart Disease Prediction App](https://heartdiseaseproject-p8wamssgyvehk3ft9ztswx.streamlit.app/)  


## 📂 Project Structure  

Heart_Disease_Project/  
│  
├── data/                  # Raw & processed datasets  
│   ├── heart_disease.csv  
│   ├── heart_disease_cleaned.csv  
│   ├── heart_disease_reduced.csv  
│   └── pca_transformed.csv  
│  
├── models/                # Trained models & preprocessing pipelines  
│   ├── preprocessor.joblib  
│   ├── rf_best.joblib  
│   └── final_model.pkl  
│  
├── notebooks/             # Jupyter/Scripted experiments  
│   ├── 01_data_cleaning.py  
│   ├── 02_pca_analysis.py  
│   ├── 03_feature_selection.py  
│   ├── 04_supervised_learning.py  
│   ├── 05_unsupervised_learning.py  
│   ├── 06_hyperparameter_tuning.py  
│   └── 07_export_model_pipeline.py  
│  
├── ui/                    # Streamlit app  
│   └── app.py  
│  
├── requirements.txt       # Python dependencies  
└── README.md              # Project documentation  

---

## ⚙️ Features  

- **Data Preprocessing**: Missing values handling, encoding, scaling.  
- **Feature Engineering**: PCA, feature selection.  
- **Supervised Learning**: Logistic Regression, Decision Trees, Random Forest, SVM.  
- **Unsupervised Learning**: Clustering with KMeans (for analysis).  
- **Model Selection**: Hyperparameter tuning for Random Forest & SVM.  
- **Final Pipeline**: Combined preprocessor + best model into a deployable pipeline.  
- **Deployment**: Interactive **Streamlit app**.  

---

## 🧠 Model Performance  

| Model                | Accuracy | Precision | Recall  | F1-score | ROC-AUC |
|----------------------|----------|-----------|---------|----------|---------|
| Logistic Regression  | 0.869    | 0.877     | 0.869   | 0.869    | 0.945   |
| Decision Tree        | 0.705    | 0.718     | 0.705   | 0.704    | 0.711   |
| Random Forest (Best) | 0.885    | 0.897     | 0.885   | 0.885    | 0.953   |
| SVM                  | 0.836    | 0.844     | 0.836   | 0.836    | 0.936   |

✅ **Best Model:** Random Forest with **Accuracy = 88.5%** and **ROC-AUC = 0.953**  

---

## 🚀 Deployment  

You can run the project locally or deploy it to **Streamlit Cloud**.  
