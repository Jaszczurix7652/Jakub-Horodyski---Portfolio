import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Perceptron:   
    def __init__(self, eta=0.10, epochs=50, is_verbose = False):       
        self.eta = eta #współczynnik uczenia
        self.epochs = epochs # ile razy perceptron będzie sie uczył
        self.is_verbose = is_verbose #czy gada
        self.list_of_errors = [] #ilosc pomyłek 
        
        
    def predict(self, x): #x - wektor cech  
        
        z = np.dot(x, self.w) #iloczyn skalarny            
        y_pred = np.sign(z)             
        return y_pred
    
    
    def fit(self, X, y): #metoda ćwicząca, X-macierz cech, y-wektor szukany    
        self.list_of_errors = [] 
        ones = np.ones((X.shape[0], 1)) #macierz jedynek
        X_1 = np.append(X.copy(), ones, axis=1) #X z doklejonymi jedynkami bo z wzoru tak wychodzi
        self.w = np.random.rand(X_1.shape[1])  #wektor wag  
        for e in range(self.epochs):  #pętla uczenia
            number_of_errors = 0
            TP = 0
            TN = 0
            FP = 0
            FN = 0 
            for x, y_target in zip(X_1,y): #pętla przejscia prze wszystkie próbki, X- próbka terningowa, y_target - znany wynik  
                y_pred = self.predict(x)
                delta_w = self.eta * (y_target - y_pred) * x #z wzoru
                self.w += delta_w                            #z wzoru 
                number_of_errors += 1 if y_target != y_pred else 0 #sprawdzanie błędów i dodawanie
                if y_target == 1 and y_pred == 1:
                    TP+=1
                if y_target == -1 and y_pred == -1:   
                    TN+=1 
                if y_target == -1 and y_pred == 1: 
                    FP+=1
                if y_target == 1 and y_pred == -1:
                    FN+=1 
              
            self.list_of_errors.append(number_of_errors)
            
            if(self.is_verbose):
                print("Próba: {}, Wagi: {}, Ilosc błędów: {}".format(
                        e+1, self.w, number_of_errors),'\n')
        print('czulosc: ', (TP/(TP+FN))*100,'%')
        print('swoistosc: ', (TN/(FP+TN))*100,'%')
        print('precyzja: ', (TP/(TP+FP))*100,'%')
        print('dokladnosc: ', ((TP+TN)/len(y))*100,'%')
        confusion_matrix = np.array([[TN, FP], [FN, TP]])
        print("Macierz pomyłek:\n", confusion_matrix)




####################################################################################################################################
attributes = ['Variance','Skewness','Curtosis', 'Entropy', 'Class']
df = pd.read_csv(r'C:\Users\Jaszc\OneDrive\Semestr 1\Algorytmy uczenia maszynowego P\perceptron_zadanie\data_banknote_authentication.txt', sep=",", header=None, names= attributes)
df=df.sample(frac=1)
X=np.array(df.iloc[:, :4])
y=np.array(df.iloc[:,4])


X_train=X[:823]            #zbiór treningowy X
X_validation=X[824:]     #zbiór walidacyjny X

ones = np.ones((X_validation.shape[0], 1)) #macierz jedynek
X_2 = np.append(X_validation.copy(), ones, axis=1)


y_train=y[:823]            #zbiór treningowy Y
y_validation=y[824:]     #zbiór walidacyjny Y

y[y==0]=-1              #zamiana 0 na -1
perceptron = Perceptron(eta=0.001, epochs=200, is_verbose = True) 

perceptron.fit(X_train,y_train)
y_pred = perceptron.predict(X_2)


wyniki=np.array([['Perceptron uzyskał prawdziwie ujemny wynik','Perceptron uzyskał fałszywie dodatni wynik'],['Perceptron uzystkał fałszywie ujemny wynik','Perceptron uzyskał prawdziwie dodatni wynik']])


%matplotlib inline
 
plt.scatter(range(perceptron.epochs), perceptron.list_of_errors)


