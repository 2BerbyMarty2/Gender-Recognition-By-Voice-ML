# Implementing K-Nearest Neighbors (KNN) for Voice Gender Recognition

While your project currently uses a Decision Tree Classifier, K-Nearest Neighbors (KNN) is another excellent algorithm to try. Because KNN relies on distance metrics to classify data points, it's particularly important to **scale your features** before training the model.

Here is a step-by-step guide to adding KNN to your project.

## 1. Import Required Libraries

You'll need `KNeighborsClassifier` for the model and `StandardScaler` to normalize your acoustic features.

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
```

## 2. Load and Prepare the Data

Assuming your dataset is loaded into a Pandas DataFrame `df`:

```python
# Load data (replace with your actual file path if different)
df = pd.read_csv('voice.csv')

# Separate features (X) and target label (y)
X = df.drop('label', axis=1)
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 3. Feature Scaling (Crucial for KNN)

Since KNN calculates the distance between points, features with larger ranges (like `meanfun` vs `sp.ent`) will unfairly dominate the distance calculation if not scaled.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 4. Train the KNN Model

```python
# Initialize KNN with a default of 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model on the scaled training data
knn.fit(X_train_scaled, y_train)
```

## 5. Evaluate the Model

```python
# Predict and evaluate
y_pred = knn.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

## 6. Hyperparameter Tuning with GridSearchCV

Just as you optimized the Decision Tree, you can optimize KNN by finding the best number of neighbors (`n_neighbors`) and distance `metric`.

```python
# Create a pipeline that scales the data first, then runs KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Note: When using a pipeline, you prefix hyperparameters with the step name + two underscores (e.g., knn__)
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

# Pass the unscaled training data! The pipeline handles scaling securely inside the cross-validation loops.
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Score:", grid_search.best_score_)
```