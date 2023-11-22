import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QComboBox, QGridLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

tipo_arvore = "Árvore de decisão"

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Classificador de risco de parada cardíaca')
        self.setGeometry(300, 300, 400, 300)

        modelo_cb = QComboBox(self)
        modelo_cb.addItem("Árvore de decisão")
        modelo_cb.addItem("Floresta aleatória")
        modelo_cb.currentTextChanged.connect(self.on_combobox_changed)
        print(modelo_cb.currentText())

        # Create 4 buttons
        button1 = QPushButton('Prever um modelo', self)
        button2 = QPushButton('Estatísticas do modelo', self)
        button3 = QPushButton('Visualizar modelo', self)
        button4 = QPushButton('Button 4', self)

        opcoes = QGridLayout(self)
        opcoes.addWidget(button1, 1, 0)
        opcoes.addWidget(button2, 1, 1)
        opcoes.addWidget(button3, 2, 0)
        opcoes.addWidget(button4, 2, 1)

        # Align buttons to the left
        layout = QGridLayout(self)

        layout.addWidget(modelo_cb, 0, 0)
        layout.addWidget(opcoes, 1, 0)

        widget = QWidget()
        self.setLayout(layout)
        self.centralWidget = widget


    def on_combobox_changed(self, value):
        tipo_arvore = value
        print(tipo_arvore)
        # do your code


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())