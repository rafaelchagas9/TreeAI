from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import pandas as pd

data: pd.DataFrame = pd.read_csv('heart_attack_data.csv')
X_train, X_test, y_train, y_test = train_test_split(data.drop('heart_attack_risk', axis=1), data['heart_attack_risk'], test_size=0.2)

# Instantiate the decision tree classifier
clf = DecisionTreeClassifier()

# Train the model on the training data
clf.fit(X_train, y_train)

# Set different decision thresholds
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Calculate recall scores for each threshold
recall_scores = []
for threshold in thresholds:
    clf.threshold = threshold
    predictions = clf.predict(X_test)
    recall = recall_score(y_test, predictions, average='weighted')
    recall_scores.append(recall)

# Identify the threshold with the highest recall
best_threshold = thresholds[recall_scores.index(max(recall_scores))]

# Set the optimal threshold for the model
clf.threshold = best_threshold
print(best_threshold)
