import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
# from graphviz import Source
# from sklearn import tree
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


dtc = DecisionTreeClassifier(max_depth=None, random_state=1, min_weight_fraction_leaf=0.5)
dtc.fit(X_train, y_train)   
pred=dtc.predict(X_validation)


dtc.X_train = X_train
dtc.y_train = y_train
dtc.X_test = X_validation
dtc.y_test = y_validation

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


# plt.figure(figsize=(12, 8))
# plot_tree(dtc, feature_names=attributes[1:17], class_names=['1', '2', '3', '4', '5', '6', '7'],
#           filled=True, rounded=True)
# plt.show()


# dot_data = tree.export_graphviz(dtc, out_file=None, feature_names=X_train.columns, class_names=['0', '1'], rounded=True, filled=True)
# graph = Source(dot_data)
# graph.render("tree", view=True, format='png',  directory='.', cleanup=True)
