from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Load your data
data = pd.read_csv('heart.csv')
features = data.drop('HeartDisease', axis=1)
target = data['HeartDisease']
features = pd.get_dummies(features)

# Split your data
from sklearn.model_selection import train_test_split
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.3, random_state=42
)

# Set the hyperparameter grid
param_grid = {
    'n_estimators': np.arange(100, 500, 50),
    'max_depth': np.arange(2, 20, 2),
    'min_samples_split': np.arange(2, 20, 2),
    'class_weight': ['balanced', 'balanced_subsample'],
    'min_samples_leaf': np.arange(2, 10, 1)
}

# Create a RandomForestClassifier
model = RandomForestClassifier()

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model
grid_search.fit(features_train, target_train)

# Get the best parameters
print("Best parameters found: ", grid_search.best_params_)
