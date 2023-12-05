import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sn
from graphviz import Source

# Set the seed value
np.random.seed(42)

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

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    df_cfm = pd.DataFrame(cm, index = ["0", "1", "2"], columns = ["0", "1", "2"])
    plt.figure(figsize = (10,7))

    cfm_plot: plt.Axes = sn.heatmap(df_cfm, annot=True)
    cfm_plot.set(xlabel='PREVISÃO', ylabel='VERDADE')
    cfm_plot.figure.savefig("cfm.png")

    # Calcula a acurácia (TP + TN) / (TP + TN + FP + FN)
    accuracy = accuracy_score(y_test, y_pred)

    print('Acurácia:', accuracy, end="")

    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    print(', Precisão:', precision, end="")

    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    print(', Recall:', recall)


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

    data = [[40,0,125,80,220,1,1]]
    df = pd.DataFrame(data, columns=['age','sex','systolic_blood_pressure','diastolic_blood_pressure','cholesterol','smoking','diabetes'])
    predict(clf, df)

    #plt.figure()
    #tree.plot_tree(clf,filled=True)  
    #plt.savefig('tree.png',format='png',bbox_inches = "tight")

    graph = Source( tree.export_graphviz(clf, out_file=None, feature_names=X_train.columns))
    png_bytes = graph.pipe(format='png')
    with open('dtree_pipe.png','wb') as f:
        f.write(png_bytes)

    # Generate confusion matrix


if __name__ == "__main__":
    main()