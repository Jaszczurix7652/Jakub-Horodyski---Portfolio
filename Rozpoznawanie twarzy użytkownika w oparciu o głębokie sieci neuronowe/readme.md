Celem projektu było zaimplementowanie głebokiej sieci neuronowej opartej na kaskadowych detekorah Haara rozpoznających twarze użytkowników w czasie rzeczywistym.

Pierwszą warstwą jest warstwa konwolucyjna, używana do wykrywania lokalnych wzorców w danych wejściowych. Filtry konwolucyjne o rozmiarze (3, 3) są przesuwane po obrazie wejściowym, wyodrębniając cechy. Funkcja aktywacji ReLU (Rectified Linear Unit) wprowadza nieliniowość, umożliwiając sieci na naukę bardziej złożonych wzorców.

Następna warstwa wykorzystuje metodę MaxPooling2D, która zmniejsza rozmiar przestrzeni cech, redukując ilość parametrów i obliczeń. Dodatkowo pobiera największą wartość z okna o ustalonym rozmiarze.

Warstwa Flatten spłaszcza dane przestrzenne do jednowymiarowego wektora. Konieczna jest przed przekazaniem danych do warstw Dense (Fully Connected) ponieważ te warstwy wymagają jednowymiarowych danych wejściowych.

W warstwie Dense każdy neuron jest połączony z każdym neuronem z poprzedniej warstwy. Posiada 64 neurony z funkcją aktywacji ReLU, co pozwala na naukę nieliniowych zależności.

Ostatnia warstwa gęsto połączona z ilością neuronów równą liczbie klas w problemie klasyfikacji. Wykorzystuje funkcję aktywacji softmax, która przekształca wyjścia sieci na prawdopodobieństwa przynależności do poszczególnych klas.

Po zbudowaniu sieci konwolucyjnej model przechodzi do etapu kompilacji. W trakcie jej określany jest optymalizator, funkcja straty i metryki oceny modelu podczas treningu. Wykorzystywany jest ImageDataGenerator do generowania nowych przykładów treningowych przez zastosowanie różnych transformacji do oryginalnych obrazów. Model jest trenowany na danych treningowych z użyciem augmentacji danych, a wyniki są monitorowane na zestawie walidacyjnym. W celu wizualizacji wykresów dokładności i funkcji straty wykorzystywana jest biblioteka matplotlib. Wytrenowany model jest zapisywany do pliku "FR.h5" w celu późniejszego użycia bez konieczności ponownego trenowania.

Wyniki modelu:
![image](https://github.com/user-attachments/assets/fc60832e-b995-4146-b680-e93c5483ce22)
