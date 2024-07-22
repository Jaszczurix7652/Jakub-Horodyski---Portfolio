Koncepcja SPC opiera się na zastosowaniu próbkowania sprzężonego, w którym piksele obrazu są pobierane w sposób nielosowy, ale zgodny z odpowiednio zaprojektowanym wzorcem próbkowania. Zamiast rejestrować pełny obraz na raz, SPC pobiera szereg pojedynczych próbek, a następnie na podstawie tych próbek dokonuje rekonstrukcji całego obrazu.
Działanie SPC można podzielić na kilka etapów. Pierwszym krokiem jest przygotowanie wzorca próbkowania, który określa sposób, w jaki piksele będą pobierane. Wzorzec ten może mieć różne formy, takie jak wzorzec losowy, pseudolosowy lub deterministyczny. Wzorzec próbkowania jest istotny dla procesu rekonstrukcji obrazu, ponieważ wpływa na jakość odtworzenia.
Kolejnym etapem jest akwizycja próbek pikseli. W tym przypadku, zamiast rejestrować pełne piksele, SPC pobiera pojedyncze wartości pikseli, które są zależne od wzorca próbkowania. Próbkowanie jest realizowane przez odpowiednie układy optyczne i detektor, które mierzą natężenie światła na poszczególnych punktach próbkowania.
Po pobraniu próbek następuje etap rekonstrukcji obrazu. Opiera się on na zastosowaniu odpowiednich algorytmów rekonstrukcji, które na podstawie pobranych próbek pikseli
odtwarzają pełny obraz. Algorytmy te wykorzystują wiedzę o wzorcu próbkowania i modelach matematycznych do odtworzenia brakujących informacji o pikselach.

Opis poszczególnych kroków algorytmu:

1. Inicjalizacja FastAPI:
Program inicjalizuje framework FastAPI na początku swojego działania. To narzędzie umożliwia obsługę interakcji z interfejsem API, ułatwiając komunikację między frontendem a backendem. Importuje niezbędne biblioteki, takie jak numpy, cv2, PIL, a także FastAPI i inne powiązane z nim elementy. Tworzy instancję aplikacji FastAPI, która posłuży do obsługi żądań przychodzących z frontendu.

2. Ustawienie Middleware CORS:
Aplikacja konfiguruje middleware CORS (Cross-Origin Resource Sharing), co umożliwia bezpieczną komunikację między frontendem a backendem. Zdefiniowane są, które źródła mogą wysyłać żądania do aplikacji, co jest istotne w kontekście zabezpieczeń przeglądarki.

3. Funkcja Generująca Macierz Przesłony DCT:
Definiuje funkcję, która generuje macierz przesłony DCT o określonych wymiarach. Wartości te będą później używane do transformacji obrazu. Proces ten obejmuje utworzenie siatki i, j, obliczenie współczynnika podziałki, a następnie obliczenie wartości macierzy DCT zgodnie ze wzorem. Pierwszy wiersz macierzy DCT jest ustawiany dla zachowania ortogonalności transformacji.

4. Endpoint FastAPI do Przetwarzania Obrazu:
Tworzy endpoint, który obsługuje przesyłanie plików obrazu. Gdy użytkownik przesyła obraz, aplikacja odczytuje go, przetwarza zgodnie z ustaloną logiką (zdefiniowaną w kolejnej funkcji), a następnie zwraca przetworzone dane w formie odpowiedzi JSON.

5.Funkcja Przetwarzająca Obraz w Pętli:
Funkcja process_image_in_loop implementuje iteracyjny proces przetwarzania obrazu z wykorzystaniem transformacji DCT. W ramach tego procesu, losowo wybierane są piksele, a opcjonalnie dodawany jest szum impulsowy, zwłaszcza gdy obraz wejściowy zawiera szum. Następnie tworzona jest macierz DCT do przekształcenia wybranych pikseli. Kolejnym krokiem jest przetwarzanie i rekonstrukcja obrazu poprzez zastosowanie transformacji DCT oraz jej odwrotnego procesu. 
Algorytm dynamicznie dostosowuje próg błędu oraz liczbę próbkowanych pikseli, co umożliwia adaptację do warunków rekonstrukcji. Proces kontynuuje się do momentu, gdy błąd średniokwadratowy między oryginalnym a odtworzonym obrazem spadnie poniżej ustalonego progu lub liczba próbkowanych pikseli nie przekroczy określonego limitu. W trakcie iteracji generowane są pliki wynikowe, takie jak obrazy z maską pikseli poddanych transformacji DCT oraz odbudowane obrazy. Warunek zakończenia zapewnia, że algorytm kończy pracę, gdy uzyska zadowalające rezultaty lub osiągnie limit próbkowania.

6. Dostosowywanie Progu Błędu i Liczby Pikseli w Próbce:
Algorytm dostosowuje próg błędu i liczbę pikseli w próbce w zależności od aktualnego błędu rekonstrukcji. Jeśli błąd jest mniejszy od progu, próg ten jest zmniejszany. Jeśli błąd jest większy niż próg błędu, liczba pikseli w próbce jest zwiększana. To dynamiczne podejście umożliwia bardziej precyzyjną rekonstrukcję obrazu w zależności od aktualnych warunków.

7. Tworzenie Obrazu z Pikseli Próbki:
Pętla iteruje po próbce pikseli, ustawiając kolor przesłony na szary (np. RGB: 101, 105, 97) w macierzy DCT. Kolor ten jest ustawiany w celu wizualizacji siatki DCT na obrazie. Następnie przetworzona próbka jest zapisywana jako obraz PNG.

8. Odtwarzanie Obrazu z Próbki Pikseli:
Transformacja DCT jest stosowana do przetworzonej próbki pikseli, a następnie wykonywana jest odwrotna transformacja DCT za pomocą transponowanej macierzy DCT. Odtworzony obraz jest zapisywany jako plik PNG.

9. Obliczanie Błędu Średniokwadratowego:
Po odtworzeniu obrazu z próbki, obliczany jest błąd średniokwadratowy między oryginalnym a odtworzonym obrazem. Błąd ten jest używany do podejmowania decyzji o dostosowywaniu progów błędu i liczby pikseli w próbce.

10. Warunek Zakończenia Algorytmu:
Algorytm kończy działanie, gdy osiągnięty zostanie ustalony próg błędu. Jest to sprawdzane za pomocą warunku, np. np.allclose, który porównuje oryginalny obraz z odtworzonym, decydując, czy różnice są akceptowalnie małe.

11. Dynamiczne Dostosowywanie Progów i Liczby Pikseli:
Algorytm dynamicznie dostosowuje próg błędu w zależności od aktualnego błędu rekonstrukcji. Jeśli błąd jest mniejszy od progu, próg ten jest zmniejszany. Jeśli błąd jest większy niż próg błędu, liczba pikseli w próbce jest zwiększana. To elastyczne podejście pozwala na adaptację do warunków rekonstrukcji.

Proces rekonstrukcji obrazu:

![image](https://github.com/user-attachments/assets/1dfd8d17-bff4-47c2-86fd-9462ee72384a)


Przykładowy wynik:

![image](https://github.com/user-attachments/assets/d544b9e6-2981-417c-9692-4f3564bee864)

