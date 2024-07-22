Badanym problemem jest analiza wpływu jakości diamentu na jego cenę sprzedaży. By wykazać zależność ceny od indywidualnych cech kamieni, zostały przeanalizowane takie właściwości jak: ilość karatów (masa kamienia 1ct = 0,2g), jakość szlifu (określa proporcje i kształt diamentu, wyznaczana jest za pomocą ocen), kolor (najcenniejsze są bezbarwne, lecz diamenty kolorowe są o wiele rzadsze co też może wpływać na cenę), czystość (odnosi się do ilości wewnętrznych zanieczyszczeń i zewnętrznych skaz takich jak pęknięcia) oraz ich wymiary.

Wykorzystywana baza danych „Diamonds” autorstwa Ulrik Thyge Pedersena zamieszczona została na portalu kaggle. Zbiór ten zawiera opis prawie 54 000 diamentów.

W celu umożliwienia dokładnej analizy wszystkich danych zawartych w bazie, zmieniono dane tekstowe na liczbowe. W przypadku tego zbioru było to możliwe do zrobienia, ponieważ przedstawione dane tekstowe oznaczają stopień w skali GIA. W tym wypadku najwyższe, najlepsze stopnie klasyfikacji oznaczone są liczbą 0, a kolejne niższe stopnie kolejnymi liczbami dodatnimi. Usunięto również wiersze, w których wartości dla kolumn „x”, „y” lub „z” wynosiły 0.

W celu predykcji ceny diamentu wykorzystano dwa modele regresji RandomForestRegressor oraz Extreme Gradient Boosting (XGBoost). Pierwszy z nich jest algorytmem opartym na drzewach decyzyjnych. Tworzy wiele drzew decyzyjnych i łączy ich wyniki, natomiast drugi to zbiorowy, bazujący na drzewach algorytm uczenia maszynowego, wykorzystujący strukturę wzmacniającą gradient. Została podjęta również próba użycia innych modeli takich jak LinearRegression, czy SVM, jednakże wyniki przez nie uzyskane były takie same lub gorsze niż te uzyskane za pomocą XGBoost i RandomForestRegressor.

Wizualizacja wyników:

![Plakat final](https://github.com/user-attachments/assets/d1a38048-7740-4c01-9aac-b122db5fb84b)
