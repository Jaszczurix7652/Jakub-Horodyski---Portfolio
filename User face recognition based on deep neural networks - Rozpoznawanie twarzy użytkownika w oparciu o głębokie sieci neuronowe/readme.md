PL:

Celem projektu było zaimplementowanie głebokiej sieci neuronowej opartej na kaskadowych detekorah Haara rozpoznających twarze użytkowników w czasie rzeczywistym.

Pierwszą warstwą jest warstwa konwolucyjna, używana do wykrywania lokalnych wzorców w danych wejściowych. Filtry konwolucyjne o rozmiarze (3, 3) są przesuwane po obrazie wejściowym, wyodrębniając cechy. Funkcja aktywacji ReLU (Rectified Linear Unit) wprowadza nieliniowość, umożliwiając sieci na naukę bardziej złożonych wzorców.

Następna warstwa wykorzystuje metodę MaxPooling2D, która zmniejsza rozmiar przestrzeni cech, redukując ilość parametrów i obliczeń. Dodatkowo pobiera największą wartość z okna o ustalonym rozmiarze.

Warstwa Flatten spłaszcza dane przestrzenne do jednowymiarowego wektora. Konieczna jest przed przekazaniem danych do warstw Dense (Fully Connected) ponieważ te warstwy wymagają jednowymiarowych danych wejściowych.

W warstwie Dense każdy neuron jest połączony z każdym neuronem z poprzedniej warstwy. Posiada 64 neurony z funkcją aktywacji ReLU, co pozwala na naukę nieliniowych zależności.

Ostatnia warstwa gęsto połączona z ilością neuronów równą liczbie klas w problemie klasyfikacji. Wykorzystuje funkcję aktywacji softmax, która przekształca wyjścia sieci na prawdopodobieństwa przynależności do poszczególnych klas.

Po zbudowaniu sieci konwolucyjnej model przechodzi do etapu kompilacji. W trakcie jej określany jest optymalizator, funkcja straty i metryki oceny modelu podczas treningu. Wykorzystywany jest ImageDataGenerator do generowania nowych przykładów treningowych przez zastosowanie różnych transformacji do oryginalnych obrazów. Model jest trenowany na danych treningowych z użyciem augmentacji danych, a wyniki są monitorowane na zestawie walidacyjnym. W celu wizualizacji wykresów dokładności i funkcji straty wykorzystywana jest biblioteka matplotlib. Wytrenowany model jest zapisywany do pliku "FR.h5" w celu późniejszego użycia bez konieczności ponownego trenowania.

Wyniki modelu:

![image](https://github.com/user-attachments/assets/fc60832e-b995-4146-b680-e93c5483ce22)


ENG:

The goal of the project was to implement a deep neural network based on cascading Haar detectors that recognize users' faces in real time.

The first layer is the convolutional layer, used to detect local patterns in the input data. Convolutional filters of size (3, 3) are moved over the input image, extracting features. The ReLU (Rectified Linear Unit) activation function introduces non-linearity, allowing the network to learn more complex patterns.

The next layer uses the MaxPooling2D method, which reduces the size of the feature space, reducing the number of parameters and calculations. Additionally, it retrieves the largest value from a window of a fixed size.

The Flatten layer flattens spatial data into a one-dimensional vector. It is necessary before passing data to Dense (Fully Connected) layers because these layers require one-dimensional input data.

In the Dense layer, each neuron is connected to each neuron from the previous layer. It has 64 neurons with the ReLU activation function, which allows for learning non-linear relationships.

The last layer is densely connected with the number of neurons equal to the number of classes in the classification problem. It uses a softmax activation function that transforms the network outputs into class membership probabilities.

After building the convolutional network, the model moves to the compilation stage. During it, the optimizer, loss function and model evaluation metrics during training are determined. ImageDataGenerator is used to generate new training examples by applying different transformations to the original images. The model is trained on training data using data augmentation and the results are monitored on the validation set. The matplotlib library is used to visualize accuracy and loss function plots. The trained model is saved to the "FR.h5" file for later use without the need to re-train.

Model results:

![image](https://github.com/user-attachments/assets/fc60832e-b995-4146-b680-e93c5483ce22)
