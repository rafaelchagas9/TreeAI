import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import tree
import matplotlib.pyplot as plt
from io import StringIO  
from graphviz import Source
from random import randint
from sklearn.tree import export_graphviz
import os

# Set the seed value
np.random.seed(42)

qnt_max_depth = 0


def train():
    global qnt_max_depth
    # Load the heart attack data
    data: pd.DataFrame = pd.read_csv('heart_attack_data.csv')

    X_train, X_test, y_train, y_test = train_test_split(data.drop('heart_attack_risk', axis=1), data['heart_attack_risk'], test_size=0.2)

    # Create a decision tree classifier
    clf = RandomForestClassifier(max_depth=qnt_max_depth)

    # Fit the model to the data
    clf.fit(X_train, y_train)
    clf.threshold = 0.2

    return X_train, X_test, y_train, y_test, clf


def measure_metrics(clf, X_test, y_test):
    """
        Calcula as métricas de precisão, acurácia e recall para um classificador.

        Acurácia é o nível geral de acertos do modelo
        Precisão é importante para minimizar o número de falsos positivos
        Recall é importante para minimizar o número de falsos negativos

        Parâmetros:
            clf (Classificador): Classificador treinado.
            X_test (array-like): Conjunto de dados de teste.
            y_test (array-like): Rótulos dos dados de teste.

        Retornos:
            accuracy (float): Acurácia do classificador.
            precision (float): Precisão do classificador.
            recall (float): Recall do classificador.
    """
    # Faz previsões
    y_pred = clf.predict(X_test)

    # Representa a taxa de acertos do modelo
    accuracy = accuracy_score(y_test, y_pred)

    # De todas as previsões feitas e positivas, quantas eram verdade?
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)

    # De todos os dados que deveriam ser previstos como verdade, quantos foram previstos corretamente?
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

    return accuracy, precision, recall


def main():
    X_train, X_test, y_train, y_test, clf = train()
    precisao, acuracia, recall = measure_metrics(clf, X_test, y_test)

    return precisao, acuracia, recall


if __name__ == "__main__":
    for j in range(0, 1):
        qnt_max_depth = 5
        n = 10

        acuracia_total = 0
        precisao_total = 0
        recall_total = 0

        for i in range(0, n):
            acuracia_atual, precisao_atual, recall_atual = main()
            acuracia_total += acuracia_atual
            precisao_total += precisao_atual
            recall_total   += recall_atual
    

        print("Acuracia: {} --- Precisão: {} --- Recall: {}".format(acuracia_total/n, precisao_total/n, recall_total/n))