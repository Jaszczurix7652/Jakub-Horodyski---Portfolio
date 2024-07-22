import pandas as pd
import numpy as np


def sigmoid(x): # Sigmoida
    return 1 / (1 + np.exp(-x))


def train_binary_classifier(X, y, eta=0.01, num_iters=1000): # Trenowanie klasyfikatora binarnego 
    m, n = X.shape
    weights = np.zeros(n+1) # Inicjalizacja wag
    X = np.hstack((np.ones((m, 1)), X)) # Dodanie kolumny z jedynkami
    
    for i in range(num_iters): # Algorytm gradientu prostego - minimalizacja funkcji kosztu
        y_pred = sigmoid(np.dot(X, weights)) # obliczanie wartosci fukcji sigmoidalnej
        error = y_pred - y                   # błąd predykcji
        gradient = np.dot(X.T, error) / m # gradient funkcji kosztu
        weights -= eta * gradient #aktualizowanie wag
    return weights


def train_one_vs_all(X, y): # Trenowanie klasyfikatora One-vs-All
    
    classifiers = {} # Pusta lista klasyfikatorów
    
    for label in np.unique(y): # Dla każdej klasy
        y_binary = np.where(y == label, 1, 0) # Stworzenie wektora etykiet binarnych
        weights = train_binary_classifier(X, y_binary) # Trenowanie klasyfikatora binarnego
        
        classifiers[label] = weights # Dodanie klasyfikatora do listy
    return classifiers


def predict_binary(X, weights): # Predykcja binarna
    X = np.hstack((np.ones((X.shape[0], 1)), X)) # Dodanie kolumny z jedynkami
    y_pred = sigmoid(np.dot(X, weights)) # Obliczenie wartości funkcji sigmoidalnej
    y_pred = np.where(y_pred >= 0.5, 1, 0) # Przekształcenie na etykietę binarną
    return y_pred


def predict_one_vs_all(X_validation, classifiers): # Predykcja dla zbioru walidacyjnego
    results = {}
    for label in classifiers:
        weights = classifiers[label]
        y_pred = predict_binary(X_validation, weights)
        results[label] = y_pred
    y_pred = pd.DataFrame(results).idxmax(axis=1)
    return y_pred


attributes = ['Name','Hair','Feathers', 'Eggs', 'Milk','Airborne', 'Aquatic','Predator','Tothed','Backbone','Breathes','Venomous','Fins','Legs','Tail','Domestic','Catsize','Type']
df=pd.read_csv(r"C:\Users\Jaszc\OneDrive\Semestr 1\Algorytmy uczenia maszynowego P\zoo.data", names=attributes )

df=df.sample(frac=1)
X=np.array(df.iloc[:, 1:17])
y=np.array(df['Type'])

X_train=X[:60]            #zbiór treningowy X
X_validation=X[61:]     #zbiór walidacyjny X


y_train=y[:60]            #zbiór treningowy Y
y_validation=y[61:]     #zbiór walidacyjny Y


classifiers = train_one_vs_all(X_train, y_train)
y_pred = predict_one_vs_all(X_validation, classifiers)


# Sprawdzanie skutecznosci
correct = 0
for i in range(len(y_validation)):
    if y_validation[i] == y_pred[i]:
        correct += 1
accuracy = correct / len(y_validation)

print('Dokladnosc: {:.2f}%'.format(accuracy*100))


wyniki=np.array([['Perceptron uzyskał prawdziwie ujemny wynik','Perceptron uzyskał fałszywie dodatni wynik'],['Perceptron uzystkał fałszywie ujemny wynik','Perceptron uzyskał prawdziwie dodatni wynik']])

# Macierz pomyłek
confusion_matrix = np.zeros((7, 7))
for i in range(len(y_validation)):
    confusion_matrix[y_validation[i]-1, y_pred[i]-1] += 1

# Obliczenie czułości, swoistości, precyzji i dokładności dla każdej klasy
def calculate_metrics(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    metrics = {'czulosc': [], 'swoistosc': [], 'precyzja': [], 'doskladnosc': []}
    for i in range(num_classes):
        TP = confusion_matrix[i,i]
        FP = np.sum(confusion_matrix[:,i])-TP
        FN = np.sum(confusion_matrix[i,:])-TP
        TN = np.sum(confusion_matrix)-TP-FP-FN
        
        czulosc = TP/(TP+FN)
        swoistosc = TN/(TN+TP)
        precyzja = TP/(TP+FP)
        doskladnosc = (TP+TN)/np.sum(confusion_matrix)
        
        metrics['czulosc'].append(czulosc)
        metrics['swoistosc'].append(swoistosc)
        metrics['precyzja'].append(precyzja)
        metrics['doskladnosc'].append(doskladnosc)
    
    return metrics

metrics = calculate_metrics(confusion_matrix)

# Wyświetlenie wyników
for i in range(len(metrics['czulosc'])):
    print(f"Klasa {i+1}:")
    print(f"Czułość: {metrics['czulosc'][i]:.2f}")
    print(f"Swoistość: {metrics['swoistosc'][i]:.2f}")
    print(f"Precyzja: {metrics['precyzja'][i]:.2f}")
    print(f"Dokładność: {metrics['doskladnosc'][i]:.2f}")
    print()



# zmodyfikować jako perceptron, to powinein byc perceptron zmodyfikowany z sigmoidą i prawdopodobienstwem



