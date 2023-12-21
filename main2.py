import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier 
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')  
X = df['review']
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Scale the data
scaler = StandardScaler(with_mean=False)  # With_mean=False because CountVectorizer output is sparse
X_train_scaled = scaler.fit_transform(X_train_vec)
X_test_scaled = scaler.transform(X_test_vec)

# Machine Learning Algorithms
algorithms = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=10000),
    'Naive Bayes': MultinomialNB(),
    'Decision Trees': DecisionTreeClassifier(random_state=42),
}

# Dummy Classifier for Trivial Baseline
dummy_classifier = DummyClassifier(strategy='most_frequent')
dummy_classifier.fit(X_train_vec, y_train)
dummy_predictions = dummy_classifier.predict(X_test_vec)
dummy_accuracy = accuracy_score(y_test, dummy_predictions)
print(f'Dummy Classifier (Majority Guess) Accuracy: {dummy_accuracy * 100:.2f}%')

# Hyperparameter Tuning using GridSearchCV for Logistic Regression
param_grid_logistic = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search_logistic = GridSearchCV(LogisticRegression(random_state=42), param_grid_logistic, cv=3, scoring='accuracy')
grid_search_logistic.fit(X_train_vec, y_train)

# Print the best hyperparameters
print("Best Hyperparameters for Logistic Regression:", grid_search_logistic.best_params_)

# Hyperparameter Tuning using GridSearchCV for Naive Bayes
param_grid_naive_bayes = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search_naive_bayes = GridSearchCV(MultinomialNB(), param_grid_naive_bayes, cv=3, scoring='accuracy')
grid_search_naive_bayes.fit(X_train_vec, y_train)

# Print the best hyperparameters
print("Best Hyperparameters for Naive Bayes:", grid_search_naive_bayes.best_params_)

# Hyperparameter Tuning using GridSearchCV for Decision Trees
param_grid_decision_tree = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_decision_tree = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_decision_tree, cv=3, scoring='accuracy')
grid_search_decision_tree.fit(X_train_vec, y_train)

# Print the best hyperparameters
print("Best Hyperparameters for Decision Trees:", grid_search_decision_tree.best_params_)

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
