PL:

Algorytm k – najbliższych sąsiadów

  Algorytm k-najbliższych sąsiadów (k-NN) to metoda uczenia maszynowego, która wykorzystuje odległości między przykładami w przestrzeni cech do przewidywania etykiet klas (klasyfikacja) lub wartości (regresja) dla nowych przykładów. Algorytm k-NN polega na znalezieniu k najbliższych sąsiadów nowego przykładu w zbiorze treningowym i opieraniu przewidywań na etykietach lub wartościach tych sąsiadów. Poniżej przedstawiona została kolejność działania algorytmu:
  
  1. Przygotowanie zbioru treningowego:
  Algorytm k-NN rozpoczyna się od posiadania zbioru treningowego, który składa się z przykładów wraz z przypisanymi im etykietami klas lub wartościami. Każdy przykład jest reprezentowany jako punkt w przestrzeni cech, gdzie każda cecha odpowiada jednej zmiennej. Ważne jest, aby dane treningowe były dobrze przygotowane, a cechy mogą wymagać normalizacji lub skalowania, aby zapewnić odpowiednią interpretację odległości.
  
  2. Wybór liczby sąsiadów (k):
  W kolejnym kroku należy wybrać liczbę sąsiadów (k), która określa, ile najbliższych sąsiadów będzie brane pod uwagę podczas klasyfikacji nowego przykładu. Wybór odpowiedniej wartości k jest ważny i może wpływać na wydajność i skuteczność algorytmu. Zbyt mała wartość k może prowadzić do nadmiernego dopasowania, podczas gdy zbyt duża wartość k może prowadzić do niedokładności.
  
  3. Obliczanie odległości:
  Dla nowego przykładu, dla którego chcemy przewidzieć etykietę klasy lub wartość, oblicza się odległość między tym przykładem a każdym przykładem w zbiorze treningowym. Najczęściej stosowaną miarą odległości jest odległość euklidesowa, która oblicza odległość między dwoma punktami jako pierwiastek sumy kwadratów różnic w wartościach cech. Istnieją także inne miary odległości, takie jak odległość Manhattan czy odległość Minkowskiego.
  4. Wybór k najbliższych sąsiadów:
  Następnie, na podstawie obliczonych odległości, wybierane są k najbliższych sąsiadów do nowego przykładu. Sąsiedztwo może być wybierane na podstawie najmniejszych odległości lub największych podobieństw, w zależności od miary odległości.
  
  5. Klasyfikacja:
  W przypadku klasyfikacji, k najbliższych sąsiadów głosuje na przynależność do konkretnej klasy, a klasa z największą liczbą głosów zostaje przewidziana dla nowego przykładu. Na przykład, kiedy liczba sąsiadów będzie wynosiła k=10, na podstawie obliczonej odległości między nowym punktem, a wszystkimi punkami w zbiorze zostanie znalezionych 10 najbliższych sąsiadów. Jeżeli spośród tych sąsiadów 5 należy do klasy A, 3 do klasy B i 1 do klasy C, to nowy punkt zostanie przypisany do klasy A, ze względu na najczęściej występującą klasę w promieniu. W sytuacji remisu pomiędzy klasami, należy zastosować dodatkowe kryteria doboru.

Perceptron One-vs-all

  Perceptron one-vs-all (jeden kontra reszta) jest algorytmem stosowanym w zadaniach klasyfikacji wieloklasowej, gdy mamy zbiór danych zawierający więcej niż dwie klasy. Polega na trenowaniu wielu perceptronów binarnych, z których każdy jest odpowiedzialny za rozróżnianie jednej klasy od pozostałych. Poniżej przedstawiona została kolejność działania algorytmu:
  
  1. Przygotowanie zbioru danych:
  Algorytm perceptronu one-vs-all rozpoczyna się podzielenia bazy danych na poszczególne zbiory. Trenowanie klasyfikatora odbywa się na zbiorze treningowym, który zawiera przykłady należące do różnych klas. Każdy przykład jest opisany zestawem cech, które są reprezentowane jako wektor. Etykiety klas są przypisane do przykładów, gdzie każda klasa ma unikalną etykietę.

  2. Inicjalizacja perceptronów:
  Dla każdej klasy w zbiorze danych, inicjalizuje się osobny perceptron binarny. Liczba perceptronów odpowiada liczbie klas w zbiorze danych.
  
  3. Trenowanie perceptronów:
  Dla każdej klasy, iteracyjnie trenuje się odpowiedni perceptron binarny. Klasa, która jest trenowana, jest traktowana jako klasa pozytywna, a pozostałe klasy są traktowane jako klasa negatywna. W trakcie treningu, perceptron jest uczony na podstawie reguły uczenia perceptronu, która polega na aktualizacji wag perceptronu w celu minimalizacji błędu klasyfikacji. Wagi perceptronu są dostosowywane na podstawie cech przykładów treningowych, a proces ten jest powtarzany do momentu, gdy perceptron osiąga zadowalającą wydajność lub zostaje osiągnięta maksymalna liczba iteracji.
  
  4. Klasyfikacja nowych przykładów:
  Po zakończeniu treningu wszystkich perceptronów, algorytm perceptronu one-vs-all może być używany do klasyfikacji nowych przykładów. Dla nowego przykładu, każdy perceptron wykonuje predykcję na podstawie obliczeń liniowych między wagami perceptronu a cechami
  nowego przykładu. Klasa przypisywana do nowego przykładu jest klasą, której perceptron osiągnął najwyższy wynik lub największe przypuszczenie.

MultiLayerPerceptron

  MultilayerPerceptron (MLP) to jedna z podstawowych architektur sztucznych sieci neuronowych. Składa się z co najmniej trzech warstw: warstwy wejściowej, jednej lub więcej warstw ukrytych oraz warstwy wyjściowej. MLP jest również znany jako sieć neuronowa z propagacją w przód. Pierwsza warstwa MLP to warstwa wejściowa, która przyjmuje dane wejściowe, np. cechy lub piksele obrazu. Każda cecha lub piksel jest reprezentowany jako jednostka wejściowa, a wartość tej jednostki odpowiada wartości cechy. MLP może zawierać jedną lub więcej warstw ukrytych między warstwą wejściową a warstwą wyjściową. Każda warstwa ukryta składa się z zestawu jednostek zwanych neuronami lub węzłami. Każdy neuron w warstwie ukrytej ma swoje wagi i oblicza ważoną sumę wejść, a następnie stosuje funkcję aktywacji, aby wygenerować wartość wyjściową. Ostatnią warstwą MLP jest warstwa wyjściowa, która generuje ostateczne wyniki predykcji lub klasyfikacji. Liczba jednostek wyjściowych w tej warstwie zależy od liczby klas lub rodzaju wyjścia, które chcemy przewidywać. Jak wcześniej wspomniano MLP działa w trybie propagacji w przód, co oznacza, że dane wejściowe przepływają przez sieć od warstwy wejściowej do warstwy wyjściowej bez żadnych cykli. Każda jednostka w warstwie ukrytej i warstwie wyjściowej oblicza ważoną sumę wejść na podstawie wag i wartości wejściowych, a następnie stosuje funkcję aktywacji, aby wygenerować wartość wyjściową. Popularne funkcje aktywacji stosowane w MLP to np. sigmoid, tangens hiperboliczny (tanh) lub funkcja ReLU (Rectified Linear Unit). 
  
  Jak w każdym innym klasyfikatorze przed rozpoczęciem klasyfikacji należy podzielić bazę danych na zbiory. Trenowanie MLP polega na dostosowywaniu wag między jednostkami, aby zminimalizować błąd predykcji. Wykorzystuje się algorytm wstecznej propagacji błędu, który oblicza gradient błędu na podstawie porównania wartości predykcji z wartościami oczekiwanymi. Wagi są aktualizowane w kierunku przeciwnym do gradientu, aby zmniejszyć błąd. Proces trenowania jest powtarzany na wielu epokach, aż sieć neuronowa osiągnie satysfakcjonującą wydajność na zbiorze treningowym.

Drzewo decyzyjne

  Drzewo decyzyjne to popularny algorytm uczenia maszynowego wykorzystywany w zadaniach klasyfikacji i regresji. Opiera się na zbudowaniu struktury drzewa, która pomaga podejmować decyzje na podstawie sekwencji warunków. Drzewo decyzyjne składa się z węzłów i krawędzi, gdzie węzły reprezentują testy na cechy, a krawędzie łączą węzły w hierarchiczną strukturę.
  
  Proces tworzenia drzewa decyzyjnego zaczyna się od korzenia, który reprezentuje cały zbiór danych treningowych. W tym węźle wybierana jest cecha, która najlepiej dzieli zbiór danych na podgrupy o różnych klasach lub wartościach celu. Istnieje wiele metryk oceny jakości podziału, takich jak indeks Giniego, entropia czy wskaźnik informacyjny Gain. Po wyborze cechy, tworzone są nowe węzły dla każdej możliwej wartości tej cechy, a dane treningowe są podzielone na podgrupy zgodnie z tym podziałem. Proces tworzenia drzewa jest rekurencyjny. Dla każdego nowo utworzonego węzła ponownie wybiera się cechę. Dla każdej podgrupy danych, które trafiają do nowego węzła, wybierana jest kolejna najlepsza cecha i tworzone są kolejne węzły. Ten proces trwa aż do spełnienia pewnych warunków zatrzymania, na przykład gdy osiągnięta zostanie maksymalna głębokość drzewa, gdy podgrupa danych jest jednorodna pod względem klasy lub gdy nie można dalej podzielić danych na podgrupy.
  
  Kiedy proces tworzenia drzewa się zakończy, liście drzewa zawierają informacje o przewidywanej klasie lub wartości celu dla danego przypadku. Gdy nowe przypadki są podawane na wejście drzewa, przechodzą one przez kolejne węzły, aż dotrą do liścia. W liściu jest dokonywana predykcja na podstawie większości klas lub średniej wartości celu w przypadkach treningowych, które tam trafiły. Drzewa decyzyjne mają tendencję do dopasowywania się do danych treningowych i mogą być podatne na przeuczenie. Pruning to proces redukcji drzewa poprzez usunięcie niektórych węzłów lub połączenie pewnych gałęzi w celu poprawy ogólnej wydajności na danych testowych.

Maszyna wektorów nośnych (SVM)

  Głównym celem SVM jest znalezienie optymalnej hiperpłaszczyzny lub zestawu hiperpłaszczyzn w przestrzeni cech, które skutecznie rozdzielają przykłady należące do różnych klas lub oszacowują wartość celu w przypadku regresji. Klasyfikator SVM ma wiele zalet, takich jak zdolność do radzenia sobie z danymi o wysokiej wymiarowości i umiejętność generalizacji na nieznanych danych. Hiperpłaszczyzna to płaszczyzna o wymiarze o jeden mniejszym niż przestrzeń cech. W przypadku dwóch klas, SVM znajduje płaszczyznę, która maksymalizuje odległość między najbliższymi punktami obu klas (wektorami nośnymi) - tę odległość nazywamy marginesem. W przypadku, gdy dane nie są liniowo separowalne, SVM używa funkcji jądra do transformacji przestrzeni cech na wyższy wymiar, gdzie dane mogą być separowalne liniowo. Popularnymi funkcjami jądra są liniowe, wielomianowe i sigmoidalne. Wybór odpowiedniej funkcji jądra zależy od charakterystyki danych i problemu. Klasyfikator
SVM stara się znaleźć hiperpłaszczyznę, która maksymalizuje margines między dwiema klasami.

  Klasyfikator SVM formułuje problem optymalizacyjny jako zadanie minimalizacji funkcji celu, której celem jest minimalizacja błędu klasyfikacji i jednoczesne maksymalizowanie marginesu. Problem optymalizacyjny jest zazwyczaj nieliniowy i convex, a do jego rozwiązania stosuje się różne metody optymalizacyjne, takie jak metoda Lagrange'a, dualność SVM i optymalizacja gradientowa. W klasyfikatorze SVM istnieje parametr regularyzacji oznaczony jako C. Wartość ta kontroluje kompromis między dopasowaniem do danych treningowych a utrzymaniem szerokiego marginesu. Dla dużych wartości C, klasyfikator SVM będzie bardziej wrażliwy na pojedyncze punkty odstające i dopasuje się dokładniej do danych treningowych. Dla mniejszych wartości C, klasyfikator SVM będzie dążył do większego marginesu kosztem dopasowania do danych treningowych. Klasyfikator SVM potrafi rozwiązywać problemy wieloklasowe korzystając z metody one-vs-rest lub one-vs-one.



ENG:

The k-nearest neighbors algorithm

  The k-nearest neighbors (k-NN) algorithm is a machine learning method that uses the distances between examples in the feature space to predict class labels (classification) or values (regression) for new examples. The k-NN algorithm involves finding the k nearest neighbors of a new example in the training set and basing predictions on the labels or values of those neighbors. The following is the sequence of operation of the algorithm:
  
  1. preparing the training set:
  The k-NN algorithm begins by having a training set, which consists of examples with their associated class labels or values. Each example is represented as a point in the feature space, where each feature corresponds to one variable. It is important that the training data is well prepared, and the features may need to be normalized or scaled to ensure that the distance is properly interpreted.
  
  2. Select the number of neighbors (k):
  In the next step, select the number of neighbors (k), which determines how many nearest neighbors will be considered when classifying a new example. Choosing the right value of k is important and can affect the efficiency and effectiveness of the algorithm. Too small a value of k can lead to over-fitting, while too large a value of k can lead to inaccuracy.

  3. Distance calculation:
  For a new example for which we want to predict a class label or value, the distance between that example and each example in the training set is calculated. The most commonly used distance measure is the Euclidean distance, which calculates the distance between two points as the root of the sum of the squares of the differences in feature values. There are also other distance measures, such as Manhattan distance or Minkowski distance.

  4. Selection of k nearest neighbors:
  Next, based on the calculated distances, the k nearest neighbors are selected for a new example. Neighbors can be selected on the basis of smallest distances or greatest similarities, depending on the distance measure.
  
  5. Classification:
  In the case of classification, the k nearest neighbors vote to belong to a particular class, and the class with the highest number of votes is provided for the new example. For example, when the number of neighbors is k=10, 10 nearest neighbors will be found based on the calculated distance between the new point and all points in the set. If among these neighbors, 5 belong to class A, 3 to class B and 1 to class C, the new point will be assigned to class A, due to the most frequent class in the radius. If there is a tie between classes, additional selection criteria should be applied.


One-vs-all perceptron

  The one-vs-all (one versus the rest) perceptron is an algorithm used in multi-class classification tasks when we have a dataset containing more than two classes. It involves training multiple binary perceptrons, each of which is responsible for distinguishing one class from the others. The order in which the algorithm works is shown below:
  
  1. preparing the dataset:
  The one-vs-all perceptron algorithm begins by dividing the database into individual sets. Training of the classifier is carried out on the training set, which contains examples belonging to different classes. Each example is described by a set of features, which are represented as a vector. Class labels are assigned to the examples, where each class has a unique label.

  2. Initialization of perceptrons:
  For each class in the dataset, a separate binary perceptron is initialized. The number of perceptrons corresponds to the number of classes in the dataset.

  3. Perceptron training:
   For each class, an appropriate binary perceptron is trained iteratively. The class that is trained is treated as a positive class and the remaining classes are treated as a negative class. During training, the perceptron is trained based on the perceptron learning rule, which consists in updating the perceptron weights to minimize the classification error. The perceptron weights are adjusted based on the characteristics of the training examples, and this process is repeated until the perceptron achieves satisfactory performance or the maximum number of iterations is reached.
  
   4. Classification of new examples:
   Once all perceptrons have been trained, the one-vs-all perceptron algorithm can be used to classify new examples. For a new example, each perceptron makes a prediction based on a linear computation between the perceptron weights and the features
new example. The class assigned to the new example is the class whose perceptron achieved the highest score or highest guess.


MultiLayerPerceptron

 MultilayerPerceptron (MLP) is one of the basic architectures of artificial neural networks. It consists of at least three layers: an input layer, one or more hidden layers, and an output layer. MLP is also known as forward propagation neural network. The first layer of MLP is the input layer, which accepts input data, e.g., image features or pixels. Each feature or pixel is represented as an input unit, and the value of this unit corresponds to the value of the feature. MLP may contain one or more hidden layers between the input layer and the output layer. Each hidden layer consists of a set of units called neurons or nodes. Each neuron in the hidden layer has its own weights and computes a weighted sum of the inputs and then applies an activation function to generate an output value. The last layer of MLP is the output layer, which generates the final prediction or classification results. The number of output units in this layer depends on the number of classes or type of output we want to predict. As previously mentioned, MLP works in forward propagation mode, which means that the input data flows through the network from the input layer to the output layer without any cycles. Each unit in the hidden layer and output layer calculates a weighted sum of inputs based on the weights and input values, and then applies an activation function to generate the output value. Popular activation functions used in MLP include sigmoid, hyperbolic tangent (tanh) or the ReLU (Rectified Linear Unit) function.

 As in any other classifier, before starting classification, the database must be divided into sets. MLP training involves adjusting weights between units to minimize prediction error. An error backpropagation algorithm is used, which calculates the error gradient based on the comparison of the prediction values ​​with the expected values. The weights are updated in the opposite direction of the gradient to reduce error. The training process is repeated over many epochs until the neural network achieves satisfactory performance on the training set.


Decision tree

 Decision tree is a popular machine learning algorithm used in classification and regression tasks. It is based on building a tree structure that helps make decisions based on a sequence of conditions. A decision tree consists of nodes and edges, where the nodes represent feature tests and the edges connect the nodes in a hierarchical structure.

 The process of creating a decision tree starts with the root, which represents the entire training data set. This node selects the feature that best divides the dataset into subgroups with different classes or goal values. There are many metrics for assessing the quality of a split, such as the Gini index, entropy, and the Gain information index. After selecting a feature, new nodes are created for each possible value of that feature, and the training data is divided into subgroups according to this division. The tree creation process is recursive. For each newly created node, a feature is selected again. For each subset of data that enters a new node, the next best feature is selected and additional nodes are created. This process continues until certain stopping conditions are met, such as when the maximum tree depth is reached, when a subgroup of data is class-homogeneous, or when the data cannot be further subgrouped.

 When the tree creation process is complete, the leaves of the tree contain information about the predicted class or target value for that case. As new cases are fed into the tree, they pass through subsequent nodes until they reach a leaf. In the leaf, a prediction is made based on the majority of the classes or the average value of the target in the training cases that ended up there. Decision trees tend to adjust to training data and may be susceptible to overfitting. Pruning is the process of reducing a tree by removing some nodes or combining some branches to improve the overall performance on test data.

Support vector machine (SVM)

 The main goal of SVM is to find an optimal hyperplane or set of hyperplanes in the feature space that effectively separates examples belonging to different classes or estimates the value of the objective in the case of regression. The SVM classifier has many advantages, such as the ability to deal with high-dimensional data and the ability to generalize on unknown data. A hyperplane is a plane with a dimension one smaller than the feature space. In the case of two classes, SVM finds a plane that maximizes the distance between the closest points of both classes (support vectors) - this distance is called the margin. In case the data is not linearly separable, SVM uses kernel functions to transform the feature space to a higher dimension where the data can be linearly separable. Popular kernel functions are linear, polynomial, and sigmoidal. The choice of an appropriate kernel function depends on the characteristics of the data and the problem. Classifier
SVM tries to find a hyperplane that maximizes the margin between two classes.

 The SVM classifier formulates the optimization problem as an objective function minimization task, the goal of which is to minimize the classification error and simultaneously maximize the margin. The optimization problem is usually nonlinear and convex, and various optimization methods such as Lagrangian method, SVM duality, and gradient optimization are used to solve it. In the SVM classifier, there is a regularization parameter denoted C. This value controls the trade-off between fitting to the training data and maintaining a wide margin. For large values ​​of C, the SVM classifier will be more sensitive to single outliers and will fit the training data more closely. For smaller values ​​of C, the SVM classifier will strive for a larger margin at the expense of fitting the training data. The SVM classifier can solve multi-class problems using the one-vs-rest or one-vs-one method.
 
  
  

  

