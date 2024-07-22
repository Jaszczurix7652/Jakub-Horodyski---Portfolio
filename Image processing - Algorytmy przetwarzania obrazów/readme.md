PL:

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



ENG:

The SPC concept is based on the use of coupled sampling, in which image pixels are taken in a non-random manner, but according to a properly designed sampling pattern. Instead of capturing the full image at once, SPC takes a series of individual samples and then reconstructs the entire image based on these samples.
The operation of SPC can be divided into several steps. The first step is to prepare a sampling pattern, which determines how the pixels will be sampled. This pattern can take various forms, such as a random pattern, pseudo-random pattern or deterministic pattern. The sampling pattern is important for the image reconstruction process because it affects the quality of the restoration.
The next step is the acquisition of pixel samples. In this case, instead of recording full pixels, the SPC takes individual pixel values that depend on the sampling pattern. Sampling is carried out by appropriate optics and a detector that measure the light intensity at each sampling point.
After sampling, the image reconstruction stage follows. It is based on the use of appropriate reconstruction algorithms that, based on the sampled pixels
reconstruct the full image. These algorithms use knowledge of the sampling pattern and mathematical models to reconstruct the missing pixel information.

1. FastAPI initialization:
The program initializes the FastAPI framework at the beginning of its operation. This tool enables it to handle API interactions, facilitating communication between the frontend and backend. It imports the necessary libraries, such as numpy, cv2, PIL, as well as FastAPI and other related components. Creates an instance of the FastAPI application, which will be used to handle incoming requests from the frontend.

2. Middleware CORS setup:
The application configures middleware CORS (Cross-Origin Resource Sharing), which enables secure communication between the frontend and backend. It defines which sources can send requests to the application, which is important in terms of browser security.

3. DCT Aperture Matrix Generating Function:
Defines a function that generates a DCT aperture matrix with specified dimensions. These values will later be used for image transformation. This process involves creating an i, j grid, calculating the pitch factor, and then calculating the DCT matrix values according to the formula. The first row of the DCT matrix is set for the orthogonality of the transformation.

4. Endpoint FastAPI for Image Processing:
Creates an endpoint that handles the upload of image files. When a user uploads an image, the application reads the image, processes it according to the established logic (defined in the next function), and then returns the processed data as a JSON response.

5. Function Processing Image in Loop:
The process_image_in_loop function implements an iterative process for processing an image using a DCT transformation. As part of this process, random pixels are selected and optional impulse noise is added, especially when the input image contains noise. A DCT matrix is then created to transform the selected pixels. The next step is to process and reconstruct the image by applying the DCT transformation and its inverse process. 
The algorithm dynamically adjusts the error threshold and the number of sampled pixels to adapt to the reconstruction conditions. The process continues until the mean-square error between the original and reconstructed images falls below a set threshold or the number of sampled pixels does not exceed a certain limit. During the iteration, result files are generated, such as DCT-transformed pixel mask images and reconstructed images. The termination condition ensures that the algorithm terminates when it obtains satisfactory results or reaches the sampling limit.

6. Adjusting the Error Threshold and Number of Pixels in the Sample:
The algorithm adjusts the error threshold and the number of pixels in the sample according to the current reconstruction error. If the error is less than the threshold, the threshold is reduced. If the error is greater than the error threshold, the number of pixels in the sample is increased. This dynamic approach allows more accurate image reconstruction depending on current conditions.

7. Creating an Image from Sample Pixels:
The loop iterates over a sample of pixels, setting the aperture color to gray (e.g., RGB: 101, 105, 97) in the DCT matrix. This color is set to visualize the DCT grid in the image. The processed sample is then saved as a PNG image.

8. Recreating the Pixel Sample Image:
A DCT transformation is applied to the processed pixel sample, and then an inverse DCT transformation is performed using the transposed DCT matrix. The reconstructed image is saved as a PNG file.

9. Calculation of Mean Square Error:
After reconstructing an image from a sample, the mean-square error between the original and reconstructed image is calculated. This error is used to make decisions about adjusting the error thresholds and the number of pixels in the sample.

10. Algorithm Termination Condition:
The algorithm terminates when a predetermined error threshold is reached. This is checked using a condition, e.g.allclose, which compares the original image with the reconstructed one, deciding whether the differences are acceptably small.

11. Dynamic Adjustment of Thresholds and Pixel Count:
The algorithm dynamically adjusts the error threshold depending on the current reconstruction error. If the error is less than the threshold, the threshold is reduced. If the error is greater than the error threshold, the number of pixels in the sample is increased. This flexible approach allows adaptation to reconstruction conditions.

Image reconstruction process:

![image](https://github.com/user-attachments/assets/1dfd8d17-bff4-47c2-86fd-9462ee72384a)


Sample result:

![image](https://github.com/user-attachments/assets/d544b9e6-2981-417c-9692-4f3564bee864)
