import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

     
attributes = ['Name','Hair','Feathers', 'Eggs', 'Milk','Airborne', 'Aquatic','Predator','Tothed','Backbone','Breathes','Venomous','Fins','Legs','Tail','Domestic','Catsize','Type']
df=pd.read_csv(r"C:\Users\Jaszc\OneDrive\Semestr 1\Algorytmy uczenia maszynowego P\zoo.data", names=attributes )

df=df.sample(frac=1)
X=np.array(df.iloc[:, 1:17])
y=np.array(df['Type'])

X_train=X[:60]            #zbiór treningowy X
X_validation=X[61:80]     #zbiór walidacyjny X
X_test=X[81:101]          #zbiór testowy X

y_train=y[:60]            #zbiór treningowy Y
y_validation=y[61:80]     #zbiór walidacyjny Y
y_test=y[81:101]          #zbiór testowy Y


svm = SVC(kernel='rbf', C=0.5, gamma='scale', random_state=1)
svm.fit(X_train, y_train)
pred=svm.predict(X_validation)


svm.X_train = X_train
svm.y_train = y_train
svm.X_test = X_validation
svm.y_test = y_validation

cm = confusion_matrix(pred, y_validation)
print("Confusion Matrix:")
print(cm)

acc = accuracy_score(pred, y_validation)
print((acc*100))

prec = precision_score(pred, y_validation, average='macro')
print((prec*100))

rec = recall_score(pred, y_validation, average='macro')
print((rec*100))

f1 = f1_score(pred, y_validation, average='macro')
print((f1*100))
