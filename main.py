import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
from io import StringIO  
from graphviz import Source


def train():
    # Load the heart attack data
    data: pd.DataFrame = pd.read_csv('heart_attack_data.csv')

    X_train, X_test, y_train, y_test = train_test_split(data.drop('heart_attack_risk', axis=1), data['heart_attack_risk'], test_size=0.2)

    # Create a decision tree classifier
    clf = DecisionTreeClassifier()
    # Fit the model to the data
    clf.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, clf


def measure_accuracy(clf, X_test, y_test):
    # Faz previsões
    y_pred = clf.predict(X_test)

    # Calcula a acurácia
    accuracy = accuracy_score(y_test, y_pred)

    print('Acurácia:', accuracy)


def predict(clf, patient):
    # Split the data into features and target
    predictions = clf.predict(patient)

    # Classify the new data
    if predictions == 0:
       print('Baixo risco de ataque cardíaco')
    elif predictions == 1:
        print('Médio risco de ataque cardíaco')
    else:
        print('Alto risco de ataque cardíaco')


def main():
    X_train, X_test, y_train, y_test, clf = train()
    measure_accuracy(clf, X_test, y_test)

    data = [[21,0,120,80,100,0,0]]
    df = pd.DataFrame(data, columns=['age','sex','systolic_blood_pressure','diastolic_blood_pressure','cholesterol','smoking','diabetes'])
    predict(clf, df)

    #plt.figure()
    #tree.plot_tree(clf,filled=True)  
    #plt.savefig('tree.png',format='png',bbox_inches = "tight")

    graph = Source( tree.export_graphviz(clf, out_file=None, feature_names=X_train.columns))
    png_bytes = graph.pipe(format='png')
    with open('dtree_pipe.png','wb') as f:
        f.write(png_bytes)


if __name__ == "__main__":
    main()