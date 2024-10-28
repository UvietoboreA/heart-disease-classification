# Heart Disease Prediction

This project aims to build a machine learning model to predict the presence of heart disease based on various health metrics. The model utilizes a Random Forest classifier for classification tasks.

## Dataset

The dataset used in this project is structured with the following columns:

- **Age**: Age of the patient (in years)
- **Sex**: Gender of the patient (1 = male; 0 = female)
- **ChestPainType**: Type of chest pain experienced (categorical)
- **RestingBP**: Resting blood pressure (in mm Hg)
- **Cholesterol**: Serum cholesterol in mg/dl (categorical: normal, above normal, and well above normal)
- **FastingBS**: Fasting blood sugar (1 = true; 0 = false)
- **RestingECG**: Resting electrocardiographic results (categorical)
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise induced angina (1 = yes; 0 = no)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **ST_Slope**: Slope of the peak exercise ST segment (categorical)
- **HeartDisease**: Target variable (1 = heart disease; 0 = no heart disease)

## Installation

To run this project, ensure you have the following libraries installed:

```bash
pip install pandas scikit-learn seaborn matplotlib
