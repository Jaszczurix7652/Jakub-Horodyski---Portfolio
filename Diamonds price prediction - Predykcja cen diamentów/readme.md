PL:

Badanym problemem jest analiza wpływu jakości diamentu na jego cenę sprzedaży. By wykazać zależność ceny od indywidualnych cech kamieni, zostały przeanalizowane takie właściwości jak: ilość karatów (masa kamienia 1ct = 0,2g), jakość szlifu (określa proporcje i kształt diamentu, wyznaczana jest za pomocą ocen), kolor (najcenniejsze są bezbarwne, lecz diamenty kolorowe są o wiele rzadsze co też może wpływać na cenę), czystość (odnosi się do ilości wewnętrznych zanieczyszczeń i zewnętrznych skaz takich jak pęknięcia) oraz ich wymiary.

Wykorzystywana baza danych „Diamonds” autorstwa Ulrik Thyge Pedersena zamieszczona została na portalu kaggle. Zbiór ten zawiera opis prawie 54 000 diamentów.

W celu umożliwienia dokładnej analizy wszystkich danych zawartych w bazie, zmieniono dane tekstowe na liczbowe. W przypadku tego zbioru było to możliwe do zrobienia, ponieważ przedstawione dane tekstowe oznaczają stopień w skali GIA. W tym wypadku najwyższe, najlepsze stopnie klasyfikacji oznaczone są liczbą 0, a kolejne niższe stopnie kolejnymi liczbami dodatnimi. Usunięto również wiersze, w których wartości dla kolumn „x”, „y” lub „z” wynosiły 0.

W celu predykcji ceny diamentu wykorzystano dwa modele regresji RandomForestRegressor oraz Extreme Gradient Boosting (XGBoost). Pierwszy z nich jest algorytmem opartym na drzewach decyzyjnych. Tworzy wiele drzew decyzyjnych i łączy ich wyniki, natomiast drugi to zbiorowy, bazujący na drzewach algorytm uczenia maszynowego, wykorzystujący strukturę wzmacniającą gradient. Została podjęta również próba użycia innych modeli takich jak LinearRegression, czy SVM, jednakże wyniki przez nie uzyskane były takie same lub gorsze niż te uzyskane za pomocą XGBoost i RandomForestRegressor.

Wizualizacja wyników:

![Plakat final](https://github.com/user-attachments/assets/d1a38048-7740-4c01-9aac-b122db5fb84b)

ENG:

The problem studied is the analysis of the influence of the quality of a diamond on its selling price. To show the dependence of the price on the individual characteristics of the stones, such properties as the number of carats (the weight of a stone 1ct = 0.2g), cut quality (determines the proportions and shape of the diamond, is determined by grades), color (the most valuable are colorless, but colored diamonds are much rarer which can also affect the price), clarity (refers to the amount of internal impurities and external blemishes such as cracks) and their dimensions were analyzed.

The “Diamonds” database used, authored by Ulrik Thyge Pedersen, is posted on the kaggle portal. This collection contains a description of almost 54,000 diamonds.

In order to enable accurate analysis of all the data contained in the database, textual data was changed to numerical. In the case of this collection, it was possible to do so because the text data presented denotes the grade on the GIA scale. In this case, the highest, best grades are denoted by the number 0, and the next lower grades by consecutive positive numbers. Rows where the values for the “x”, “y” or “z” columns were 0 were also removed.

Two regression models RandomForestRegressor and Extreme Gradient Boosting (XGBoost) were used to predict the price of diamond. The former is an algorithm based on decision trees. It creates multiple decision trees and combines their results, while the second is a collective tree-based machine learning algorithm using a gradient boosting structure. An attempt was also made to use other models such as LinearRegression, or SVM, however the results they produced were the same or worse than those obtained using XGBoost and RandomForestRegressor.

Visualization of results:

![Plakat final](https://github.com/user-attachments/assets/d1a38048-7740-4c01-9aac-b122db5fb84b)
