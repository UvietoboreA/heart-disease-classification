import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('heart.csv')

# Preprocess the features and target
features = data.drop('HeartDisease', axis=1)
target = data['HeartDisease']
features = pd.get_dummies(features)

# Split the data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=350, max_depth=7, class_weight='balanced_subsample', 
    max_features='sqrt', min_samples_split=14)

# Fit the model
fit_model = model.fit(features_train, target_train)

# Make predictions
predictions = fit_model.predict(features_test)

# Evaluate the model
accuracy = accuracy_score(target_test, predictions)
conf_matrix = confusion_matrix(target_test, predictions)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(target_test, predictions))

# Cross-validation score
scores = cross_val_score(model, features, target, cv=10)
print("Cross-Validation Score Mean:", scores.mean())

# Visualize the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrRd')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

importances = model.feature_importances_
print(importances)