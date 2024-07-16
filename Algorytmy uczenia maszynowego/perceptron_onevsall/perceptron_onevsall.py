import numpy as np
import pandas as pd

class Perceptron:   
    def __init__(self, eta=0.10, epochs=50, is_verbose = False):       
        self.eta = eta #współczynnik uczenia
        self.epochs = epochs # ile razy perceptron będzie sie uczył
        self.is_verbose = is_verbose #czy gada
        self.list_of_errors = [] #ilosc pomyłek 
        
        
    def predict(self, X):        #najpierw klasyfikacja a pozniej signum
        return np.dot(X, self.w) #iloczyn skalarny

    def activation_function(self, dot_probka): 
        return np.sign(dot_probka) #signum
    
    
    def fit(self, X, y): #metoda ćwicząca, X-macierz cech, y-wektor szukany    
        self.list_of_errors = [] 
        ones = np.ones((X.shape[0], 1)) #macierz jedynek
        X_1 = np.append(X.copy(), ones, axis=1) #X z doklejonymi jedynkami bo z wzoru tak wychodzi
        self.w = np.random.rand(X_1.shape[1])  #wektor wag  
        for e in range(self.epochs):  #pętla uczenia
            
            for x, y_target in zip(X_1,y): #pętla przejscia prze wszystkie próbki, X- próbka terningowa, y_target - znany wynik  
                y_pred = self.activation_function(self.predict(x))
                delta_w = self.eta * (y_target - y_pred) * x #z wzoru
                self.w += delta_w                            #z wzoru 
                
            if(self.is_verbose):
                print("Próba: {}, Wagi: {}".format(
                        e+1, self.w, ),'\n')
        return self.w
    
    
    def train_one_vs_all(self, Xo, yo): # Trenowanie klasyfikatora One-vs-All
        
        classifiers = {} # Pusta lista klasyfikatorów
        
        for label in np.unique(yo): # Dla każdej klasy
            y_binary = np.where(yo == label, 1, -1) # Stworzenie wektora etykiet binarnych
            self.w = perceptron.fit(Xo, y_binary) # Trenowanie klasyfikatora binarnego
            
            classifiers[label] = self.w # Dodanie klasyfikatora do listy
        return classifiers
    
    
    def predict_one_vs_all(self, X_validationo, classifierso): # Predykcja dla zbioru walidacyjnego
        results = []
        for label in classifierso:
            self.w = classifierso[label]
            ones = np.ones((X_validationo.shape[0], 1)) #macierz jedynek
            X_1 = np.append(X_validationo.copy(), ones, axis=1)
            y_pred = perceptron.predict(X_1)
            results.append(y_pred)
        y_pred = np.argmax(results, axis=0) #sprawdznie odleglosci
                                            #szukanie indeksu klasy o najwyższej wartości
                                            #wybiera klasę o najwyższej wartocsci jako wynik predykcji
        return y_pred+1
        
    
    # Obliczenie czułości, swoistości, precyzji i dokładności dla każdej klasy
    def calculate_metrics(self, confusion_matrix1):
        num_classes = confusion_matrix1.shape[0]
        metrics = {'czulosc': [], 'swoistosc': [], 'precyzja': [], 'doskladnosc': []}
        for i in range(num_classes):
            TP = confusion_matrix1[i,i]
            FP = np.sum(confusion_matrix1[:,i])-TP
            FN = np.sum(confusion_matrix1[i,:])-TP
            TN = np.sum(confusion_matrix1)-TP-FP-FN
            
            czulosc = TP/(TP+FN)
            swoistosc = TN/(TN+TP)
            precyzja = TP/(TP+FP)
            doskladnosc = (TP+TN)/np.sum(confusion_matrix1)
            
            metrics['czulosc'].append(czulosc)
            metrics['swoistosc'].append(swoistosc)
            metrics['precyzja'].append(precyzja)
            metrics['doskladnosc'].append(doskladnosc)
        
        return metrics



####################################################################################################################################
attributes = ['Name','Hair','Feathers', 'Eggs', 'Milk','Airborne', 'Aquatic','Predator','Tothed','Backbone','Breathes','Venomous','Fins','Legs','Tail','Domestic','Catsize','Type']
df=pd.read_csv(r"C:\Users\Jaszc\OneDrive\Semestr 1\Algorytmy uczenia maszynowego P\zoo.data", names=attributes )

df=df.sample(frac=1)
X=np.array(df.iloc[:, 1:17])
y=np.array(df['Type'])

X_train=X[:60]            #zbiór treningowy X
X_validation=X[61:80]     #zbiór walidacyjny X
X_test=X[81:101]  

y_train=y[:60]            #zbiór treningowy Y
y_validation=y[61:80]     #zbiór walidacyjny Y
y_test=y[81:101]          #zbiór testowy X

perceptron = Perceptron(eta=0.001, epochs=200, is_verbose = True) 

classifiers = perceptron.train_one_vs_all(X_train, y_train)
y_pred = perceptron.predict_one_vs_all(X_test, classifiers)

#Sprawdzanie dokładnoci
correct = 0
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        correct += 1
accuracy = correct / len(y_test)

print('Dokladnosc: {:.2f}%'.format(accuracy*100))


# Macierz pomyłek
confusion_matrix1 = np.zeros((7, 7))
for i in range(len(y_test)):
    confusion_matrix1[y_pred[i]-1, y_test[i]-1] += 1


metrics = perceptron.calculate_metrics(confusion_matrix1)



# Wyświetlenie wyników
for i in range(len(metrics['czulosc'])):
    print(f"Klasa {i+1}:")
    print(f"Czułość: {metrics['czulosc'][i]:.2f}")
    print(f"Swoistość: {metrics['swoistosc'][i]:.2f}")
    print(f"Precyzja: {metrics['precyzja'][i]:.2f}")
    print(f"Dokładność: {metrics['doskladnosc'][i]:.2f}")
    print()



from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
cm = confusion_matrix(y_pred, y_test)
print("Confusion Matrix:")
print(cm)

acc = accuracy_score(y_test, y_pred)
print("Dokladnosc: {:.2f}%".format(acc*100))

prec = precision_score(y_test, y_pred, average='macro')
print("Precyzja: {:.2f}%".format(prec*100))

rec = recall_score(y_test, y_pred, average='macro')
print("Czulosc: {:.2f}%".format(rec*100))

f1 = f1_score(y_test, y_pred, average='macro')
print("Swoistosc: {:.2f}%".format(f1*100))