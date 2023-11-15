import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

nume_fisier = 'C:\\Users\\haske\\OneDrive\\Desktop\\AI Facultate\\AI-Labs\\Tema4\\seeds_dataset.csv'
nume_coloane = ['Coloana1', 'Coloana2', 'Coloana3', 'Coloana4', 'Coloana5', 'Coloana6', 'Coloana7', 'Eticheta']
date = pd.read_csv(nume_fisier, sep=',', header=None, names=nume_coloane)

date_antrenare, date_testare = train_test_split(date, test_size=0.3, random_state=42)

dim_strat_intrare = 7
dim_strat_ascuns = 3
dim_strat_iesire = 1

rata_invatare = 0.01
nr_maxim_epoci = 10

ponderi_intrare_ascuns = np.random.randn(dim_strat_intrare, dim_strat_ascuns) * 0.01
ponderi_ascuns_iesire = np.random.randn(dim_strat_ascuns, dim_strat_iesire) * 0.01

# Afișăm ponderile inițializate
print("Ponderi strat intrare-ascuns:")
print(ponderi_intrare_ascuns)
print("\nPonderi strat ascuns-ieșire:")
print(ponderi_ascuns_iesire)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0) * 1

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def forward_propagation(input_data, weights_input_hidden, weights_hidden_output, activation_function):
    input_to_hidden = np.dot(input_data, weights_input_hidden)
    output_from_hidden = activation_function(input_to_hidden)

    input_to_output = np.dot(output_from_hidden, weights_hidden_output)

    output = activation_function(input_to_output)

    return output

X = date.iloc[:, :-1]  # Toate coloanele, mai puțin ultima
y = date.iloc[:, -1]   # Doar ultima coloană

weights_input_hidden = np.random.rand(7, 3)  # presupunem 4 neuroni în stratul ascuns
weights_hidden_output = np.random.rand(3, 1)         # presupunem o singură ieșire



for index, row in X.iterrows():
    output = forward_propagation(row, weights_input_hidden, weights_hidden_output, sigmoid)
    print(f"Ieșirea pentru rândul {index}: {output}")