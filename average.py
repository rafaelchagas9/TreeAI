import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn import tree
import matplotlib.pyplot as plt
from io import StringIO  
from graphviz import Source
from random import randint

qnt_max_depth = 0


def train():
    global qnt_max_depth
    # Load the heart attack data
    data: pd.DataFrame = pd.read_csv('heart_attack_data.csv')

    X_train, X_test, y_train, y_test = train_test_split(data.drop('heart_attack_risk', axis=1), data['heart_attack_risk'], test_size=0.2)

    # Create a decision tree classifier
    clf = RandomForestClassifier(max_depth=4)
    # Fit the model to the data
    clf.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, clf


def measure_accuracy(clf, X_test, y_test):
    # Faz previsões
    y_pred = clf.predict(X_test)

    # Calcula a acurácia
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')

    return accuracy, precision


def main():
    X_train, X_test, y_train, y_test, clf = train()
    precisao, acuracia = measure_accuracy(clf, X_test, y_test)

    return precisao, acuracia


if __name__ == "__main__":
    for j in range(0, 1):
        qnt_max_depth = 10
        n = 100
        acuracia_total = 0
        precisao_total = 0
        for i in range(0, n):
            acuracia_atual, precisao_atual = main()
            acuracia_total += acuracia_atual
            precisao_total += precisao_atual
    

        print("Precisao: {} --- Acuracia: {}".format(precisao_total/n, acuracia_total/n))