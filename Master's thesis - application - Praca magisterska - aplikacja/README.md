PL:

Do analizy porównawczej popularnych rozwiązań z autorskim modelem oraz wygodniejszej obsługi została wykonana aplikacja z intuicyjnym, graficznym interfejsem. Zwalnia ona użytkownika z konieczności wpisywania skomplikowanych komend do konsoli programu. Opis można podzielić na dwie części: back-end i front-end.

Back-end:
Głównym celem zaplecza programu (ang. Back-end) jest udostępnienie w formie API wcześniej przytoczonych przykładów wraz z autorskim modelem. Aplikacja została napisana jest w języku Python i opiera się na bibliotece FastAPI, która pozwala na integracje wielu domen i tworzenie lokalnych lub publicznych serwerów HTTP poprzez przyjmowanie oraz wysyłanie żądań z różnych źródeł. Każdy zaimplementowany model posiada własny punkt końcowy (ang. endpoint), POST lub GET. Przykładowo, DALL-E 2 przyjmuje opis obrazu wysłany przez użytkownika, generuje wynik w chmurze, a następnie zwraca go jako adres URL używając funkcji POST do serwera. Odwrotne żądanie zostało zastosowane dla Stable Diffusion v1.5, gdzie GET pobiera opis, przetwarza obraz na urządzeniu użytkownika, a następnie wysyła go w formie odpowiedzi. Autorski model nie wymaga tworzenia API, lecz jedynie importu funkcji generowania wyniku z opisu tekstowego, który jest pobierany i przesyłany na serwer.

Biblioteka FastAPI umożliwia również sprawdzenie poprawności działania programu nie posiadając utworzonego interfejsu. Jest to bardzo pomocne w początkowych etapach programowania aplikacji. W polu „prompt” można wpisać swój opis, a następnie przekazać go do generatora przyciskiem „Execute”. 

Front-end:
Front-end aplikacji został wykonany w języku programowania JavaScript, React z użyciem biblioteki Chakra UI. Celem jego jest odbieranie i wysyłanie żądań do poprzednio opisanego programu oraz obsługa interfejsu. 
W górnej części okna umieszone zostało pole do wpisywania opisu tekstowego przez użytkownika. Poniżej znajdują się przyciski, za pomocą których należy wybrać odpowiedni generator. W oczekiwaniu na wynik wyświetlana jest animacja przetwarzania, zapewniająca zainteresowanego o poprawnym wczytaniu warunku i dokonaniu wyboru. Po zakończeniu procesu, wygenerowany obraz zostaje wyświetlony poniżej przycisków.W momencie wykonania operacji w błędnej kolejności pojawia się okno z komunikatem o treści błędu, podobna sytuacja następuje przy nieprawidłowości działania programu. 


![image](https://github.com/user-attachments/assets/71b0e7b7-7745-4533-8990-3496bb187a49)


ENG:

An application with an intuitive graphical interface was created for comparative analysis of popular solutions with the original model and for more convenient operation. It frees the user from the need to enter complex commands into the program console. The description can be divided into two parts: back-end and front-end.

Back-end:
The main goal of the program's back-end is to make the previously cited examples available along with the original model in the form of an API. The application is written in Python and is based on the FastAPI library, which allows the integration of multiple domains and the creation of local or public HTTP servers by accepting and sending requests from various sources. Each implemented model has its own endpoint, POST or GET. For example, DALL-E 2 accepts an image description sent by the user, generates the result in the cloud, and then returns it as a URL using the POST function to the server. The reverse request was used for Stable Diffusion v1.5, where GET takes the description, processes the image on the user's device, and then sends it as a response. The original model does not require the creation of an API, but only the import of the function to generate the result from the text description, which is downloaded and sent to the server.

The FastAPI library also allows you to check the correct operation of the program without having to create an interface. This is very helpful in the initial stages of application development. You can enter your description in the "prompt" field and then send it to the generator by pressing the "Execute" button.

Front-end:
The front-end of the application was made in the JavaScript programming language, React, using the Chakra UI library. Its purpose is to receive and send requests to the previously described program and to operate the interface.
In the upper part of the window there is a field for entering a text description by the user. Below are the buttons to select the appropriate generator. While waiting for the result, a processing animation is displayed, assuring the interested party that the condition has been correctly loaded and the selection has been made. After the process is completed, the generated image is displayed below the buttons. When the operation is performed in the wrong order, a window appears with an error message, a similar situation occurs when the program does not operate properly.

![image](https://github.com/user-attachments/assets/71b0e7b7-7745-4533-8990-3496bb187a49)
