import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('IMDB Dataset.csv')  
X = df['review']
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
# Scale the data
scaler = StandardScaler(with_mean=False)  
X_train_scaled = scaler.fit_transform(X_train_vec)
X_test_scaled = scaler.transform(X_test_vec)

# Machine Learning Algorithms
algorithms = {
    'Logistic Regression':LogisticRegression(random_state=42, max_iter=10000),

    'Naive Bayes': MultinomialNB(),
    'Decision Trees': DecisionTreeClassifier(random_state=42),
}

# Dummy Classifier for Trivial Baseline
dummy_classifier = DummyClassifier(strategy='most_frequent')
dummy_classifier.fit(X_train_vec, y_train)
dummy_predictions = dummy_classifier.predict(X_test_vec)
dummy_accuracy = accuracy_score(y_test, dummy_predictions)
print(f'Dummy Classifier  Accuracy: {dummy_accuracy * 100:.2f}%')

# Hyperparameter Tuning using GridSearchCV for Logistic Regression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_vec, y_train)

# Print the best hyperparameters
print("Best Hyperparameters for Logistic Regression:", grid_search.best_params_)

# Evaluate the models
for name, algorithm in algorithms.items():
    if name == 'Logistic Regression':
        algorithm.fit(X_train_scaled, y_train)
        y_pred = algorithm.predict(X_test_scaled)
    else:
        # For Naive Bayes and Decision Trees, apply cross-validation
        scores = cross_val_score(algorithm, X_train_vec, y_train, cv=3, scoring='accuracy')
        print(f'{name} Cross-Validation Accuracy: {scores.mean() * 100:.2f}%')

        # Fit the model on the entire training set for reporting metrics
        algorithm.fit(X_train_vec, y_train)
        y_pred = algorithm.predict(X_test_vec)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy * 100:.2f}%')
    print(f'{name} Classification Report:\n{classification_report(y_test, y_pred)}')
    print('-' * 50)
