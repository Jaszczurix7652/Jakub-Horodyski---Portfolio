#%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import numpy

#Promień sąsiedztwa
# k=round(sqrt(len(df1)))

class KNN:
    def __init__(self, k):
        self.k=k
        
    def fit(self, x, y): #sprawdzenie długoci danych
        assert len(x) == len(y)
        self.X_train = x
        self.Y_train = y
        
    def distance(self, X1, X2):  #Euklidesowo
        X1, X2 = np.array(X1), np.array(X2)
        distance = 0             #domyslna odleglosc
        for i in range(len(X1) - 1):
            distance += (X1[i] - X2[i]) ** 2
        return np.sqrt(distance)
    
    def predict(self, X_validation):
       
        sorted_output = []
        for i in range(len(X_validation)): #petla irerujaca dlugosci walidacyjnego
            distances = []
            neighbors = []
            for j in range(len(self.X_train)): #uruchomienie długoci danych X train
                dist = self.distance(self.X_train[j], X_validation[i])
                distances.append([dist, j])
            distances.sort()                    #sortowanie odległoci
            distances = distances[0:self.k]     #ograniczenia rozwiązań do promienia k
            for _, j in distances:
                neighbors.append(self.Y_train[j])
            ans = max(neighbors)
            sorted_output.append(ans)

        return sorted_output
            
    
    
    def score(self, X_validation, Y_validation): #wynik przewidywania
        
        predictions = self.predict(X_validation)
        return [predictions, (predictions == Y_validation).sum() / len(Y_validation)]

    
    # Obliczenie czułości, swoistości, precyzji i dokładności dla każdej klasy
    def calculate_metrics(self, confusion_matrix1):
        num_classes=7
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
    



#otwieranie bazy danych
attributes = ['Name','Hair','Feathers', 'Eggs', 'Milk','Airborne', 'Aquatic','Predator','Tothed','Backbone','Breathes','Venomous','Fins','Legs','Tail','Domestic','Catsize','Type']
df=pd.read_csv(r"C:\Users\Jaszc\OneDrive\Semestr 1\Algorytmy uczenia maszynowego P\zoo.data", names=attributes )

df=df.sample(frac=1)
df1=np.array(df.iloc[:, 1:17])
df2=np.array(df['Type'])



#podział bazy danych na zbiory
X_train=df1[:60]            #zbiór treningowy X
X_validation=df1[61:80]     #zbiór walidacyjny X
X_test=df1[81:101]          #zbiór testowy X

Y_train=df2[:60]            #zbiór treningowy Y
Y_validation=df2[61:80]     #zbiór walidacyjny Y
y_test=df2[81:101]          #zbiór testowy Y

#zapis zbiorów do plików

# numpy.savetxt(r'C:\Users\Jaszc\OneDrive\Semestr 1\Algorytmy uczenia maszynowego P\X_train.csv',X_train,delimiter=",")
# numpy.savetxt(r'C:\Users\Jaszc\OneDrive\Semestr 1\Algorytmy uczenia maszynowego P\X_validation.csv',X_validation,delimiter=",")
# numpy.savetxt(r'C:\Users\Jaszc\OneDrive\Semestr 1\Algorytmy uczenia maszynowego P\X_test.csv',X_test,delimiter=",")

# numpy.savetxt(r'C:\Users\Jaszc\OneDrive\Semestr 1\Algorytmy uczenia maszynowego P\Y_train.csv',Y_train,delimiter=",")
# numpy.savetxt(r'C:\Users\Jaszc\OneDrive\Semestr 1\Algorytmy uczenia maszynowego P\Y_validation.csv',Y_validation,delimiter=",")
# numpy.savetxt(r'C:\Users\Jaszc\OneDrive\Semestr 1\Algorytmy uczenia maszynowego P\Y_test.csv',y_test,delimiter=",")



testing_neighbors = KNN(k=10)
testing_neighbors.fit(X_train, Y_train)
prediction = testing_neighbors.predict(X_validation)

accuracy = testing_neighbors.score(X_validation, Y_validation)[1]*100
print('Accuracy:', "%.2f" % accuracy,'%')


#tablica pomyłek
wyniki=np.array([[' Perceptron uzyskał prawdziwie ujemny wynik','Perceptron uzyskał fałszywie dodatni wynik'],['Perceptron uzystkał fałszywie ujemny wynik','Perceptron uzyskał prawdziwie dodatni wynik']])

y_pred = testing_neighbors.score( X_validation, Y_validation)[0]

confusion_matrix = np.zeros((7, 7))
for i in range(len(Y_validation)):
    confusion_matrix[Y_validation[i]-1, y_pred[i]-1] += 1

metrics = testing_neighbors.calculate_metrics(confusion_matrix)

# Wyświetlenie wyników
for i in range(len(metrics['czulosc'])):
    print(f"Klasa {i+1}:")
    print(f"Czułość: {metrics['czulosc'][i]:.2f}")
    print(f"Swoistość: {metrics['swoistosc'][i]:.2f}")
    print(f"Precyzja: {metrics['precyzja'][i]:.2f}")
    print(f"Dokładność: {metrics['doskladnosc'][i]:.2f}")
    print()

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
cm = confusion_matrix(y_pred, Y_validation)
print("Confusion Matrix:")
print(cm)

acc = accuracy_score(Y_validation, y_pred)
print("%.2f" %(acc*100))

prec = precision_score(Y_validation, y_pred, average='macro')
print("%.2f" %(prec*100))

rec = recall_score(Y_validation, y_pred, average='macro')
print("%.2f" %(rec*100))

f1 = f1_score(Y_validation, y_pred, average='macro')
print("%.2f" %(f1*100))
