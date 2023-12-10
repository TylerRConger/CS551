# https://obrunet.github.io/data%20science/Adult_Census_Income/

import pandas as pd
from sklearn.model_selection import train_test_split
from autosklearn.classification import AutoSklearnClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load data
data = pd.read_csv('adult/adult.data', header=None)
print(data.columns[2])
data = data.drop(data.columns[4], axis=1)
data = data.drop(data.columns[2], axis=1)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]


# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and fit the AutoML classifier
model = AutoSklearnClassifier(time_left_for_this_task=30, per_run_time_limit=1)
model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = model.predict(X_test)

# Evaluate the model
accuracy_autoML = accuracy_score(Y_test, Y_pred)
f1_autoML = f1_score(Y_test, Y_pred, average='binary', pos_label=' >50K')
classReport = classification_report(Y_test, Y_pred)

print("AutoML Results:")
print('Accuracy: ', accuracy_autoML)
print('F1-score: ', f1_autoML)
print(classReport)



print(f': on train = {model.score(X_train, Y_train)*100:.2f}%, on test = {model.score(X_test, Y_test)*100:.2f}%')

print("End Auto ML \n\n\n ===================================")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

#print(data.head())
#print(data.columns)
print(data[1])
categoricalCols = [1, 3, 5, 6, 7, 8, 9, 13, 14]
#data = pd.get_dummies(data, columns=categoricalCols)

print(data.head())

# Separate features and target variable
X = data.drop(14, axis=1)
Y = data[14]

# Encode categorical variables
encoder = LabelEncoder()
X_encoded = X.apply(encoder.fit_transform)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)

# Use randomforestclassification
pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),  # Handle missing values
    RandomForestClassifier(random_state=42)
)

# Perform manual hyperparameter tuning using GridSearchCv
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200, 500, 1000],
    'randomforestclassifier__max_depth': [None, 10, 20, 30, 40],
}

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=3)
grid_search.fit(X_train, Y_train)

# Print the best parameters from the grid search
print("Best Parameters:", grid_search.best_params_)

# Make predictions on the test set
Y_pred = grid_search.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
classification_rep = classification_report(Y_test, Y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)