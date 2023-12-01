import numpy as np
import pandas as pd
import autosklearn.classification
import pandas as pd
from sklearn.model_selection import train_test_split



data = pd.read_csv('./adult/adult.data')
print(data.head())

# Assuming the target variable is in the column 'income'
X = data.drop('income', axis=1)
y = data['income']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train AutoML model
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30)
automl.fit(X_train, y_train)

# Evaluate the model
accuracy = automl.score(X_test, y_test)
print(f'Accuracy: {accuracy}')