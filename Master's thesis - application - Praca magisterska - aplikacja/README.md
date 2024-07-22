Do analizy porównawczej popularnych rozwiązań z autorskim modelem oraz wygodniejszej obsługi została wykonana aplikacja z intuicyjnym, graficznym interfejsem. Zwalnia ona użytkownika z konieczności wpisywania skomplikowanych komend do konsoli programu. Opis można podzielić na dwie części: back-end i front-end.

Back-end:
Głównym celem zaplecza programu (ang. Back-end) jest udostępnienie w formie API wcześniej przytoczonych przykładów wraz z autorskim modelem. Aplikacja została napisana jest w języku Python i opiera się na bibliotece FastAPI, która pozwala na integracje wielu domen i tworzenie lokalnych lub publicznych serwerów HTTP poprzez przyjmowanie oraz wysyłanie żądań z różnych źródeł. Każdy zaimplementowany model posiada własny punkt końcowy (ang. endpoint), POST lub GET. Przykładowo, DALL-E 2 przyjmuje opis obrazu wysłany przez użytkownika, generuje wynik w chmurze, a następnie zwraca go jako adres URL używając funkcji POST do serwera. Odwrotne żądanie zostało zastosowane dla Stable Diffusion v1.5, gdzie GET pobiera opis, przetwarza obraz na urządzeniu użytkownika, a następnie wysyła go w formie odpowiedzi. Autorski model nie wymaga tworzenia API, lecz jedynie importu funkcji generowania wyniku z opisu tekstowego, który jest pobierany i przesyłany na serwer.

Biblioteka FastAPI umożliwia również sprawdzenie poprawności działania programu nie posiadając utworzonego interfejsu. Jest to bardzo pomocne w początkowych etapach programowania aplikacji. W polu „prompt” można wpisać swój opis, a następnie przekazać go do generatora przyciskiem „Execute”. 

Front-end:
Front-end aplikacji został wykonany w języku programowania JavaScript, React z użyciem biblioteki Chakra UI. Celem jego jest odbieranie i wysyłanie żądań do poprzednio opisanego programu oraz obsługa interfejsu. 
W górnej części okna umieszone zostało pole do wpisywania opisu tekstowego przez użytkownika. Poniżej znajdują się przyciski, za pomocą których należy wybrać odpowiedni generator. W oczekiwaniu na wynik wyświetlana jest animacja przetwarzania, zapewniająca zainteresowanego o poprawnym wczytaniu warunku i dokonaniu wyboru. Po zakończeniu procesu, wygenerowany obraz zostaje wyświetlony poniżej przycisków.W momencie wykonania operacji w błędnej kolejności pojawia się okno z komunikatem o treści błędu, podobna sytuacja następuje przy nieprawidłowości działania programu. 


![image](https://github.com/user-attachments/assets/71b0e7b7-7745-4533-8990-3496bb187a49)
