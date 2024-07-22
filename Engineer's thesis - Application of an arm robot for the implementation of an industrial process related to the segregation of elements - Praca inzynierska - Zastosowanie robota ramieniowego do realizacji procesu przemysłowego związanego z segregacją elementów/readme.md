Celem pracy jest analiza możliwości wykorzystania przemysłowego robota ramieniowego do realizacji wybranego procesu przemysłowego związanego z segregacją elementów
W zakres pracy wchodziło:
1. przystosowanie stanowiska z robotem ramieniowym do wykonania wybranego procesu technologicznego;
2. opracowanie i wykonanie stanowiska laboratoryjnego;
3. opracowanie i wykonanie programu sterującego robotem przy wykorzystaniu środowiska RT-Toolbox3.

W repozytorium został umieszczony program pomiaru rezystywności został napisany w środowisku Arduino IDE w wersji 2.0.1.

Na początku programu zostały zdefiniowane wszystkie użyte biblioteki oraz ich deklaracja wraz z „przestrzenią nazw”.

Następnie została utworzona funkcja „setup”, w której znajdują się ustawienia początkowe monitora portu szeregowego, modułu INA219, definiowanie cyfrowych portów wyjściowych oraz wyświetlanie wiadomości dla użytkownika przy każdorazowym włączeniu programu, takie jak „Hello!”, „Nie można wykryc czujnika” – kiedy moduł jest niepodłączony oraz „Rozpoczynanie pomiaru…” – kiedy moduł został uruchomiony poprawnie i program może przejść do kolejnej sekcji.

W kolejnym kroku została utworzona pętla „loop”, na której początku zostały zdefiniowane wszystkie wartości użyte do obliczeń jak i pomiaru.

Kolejną częścią kodu programu jest ustawienie wszystkich wyjść cyfrowych kontrolera na stan niski, aby przy każdym nowym uruchomieniu bity zostały zresetowane.

Następną ważną częścią jest zdefiniowanie oraz ustawienie wyświetlania wiadomości na monitorze portu szeregowego „Podaj srednice elementu w metrach: „ i „Podaj dlugosc elementu w metrach”. Wiadomości te są powtarzane co 10 sekund w celu przypomnienia. Po wyświetleniu pierwszej wiadomości użytkownik musi wpisać odpowiedni wymiar elementu, kiedy to zrobi pojawi się wczytana wartości, a zaraz po niej drugi komunikat. Dopiero po wykonaniu tych czynności program przejdzie do kolejnego etapu.

W dalszej części znajduje się obliczenie pola przekroju poprzecznego elementu segregowanego – S, na podstawie wcześniej wpisanych danych przez użytkownika.

W tej chwili program przechodzi do najważniejszej części kodu jaką jest pętla „obliczanie:”. Na początku ustawiane są stany niskie na wyjściach cyfrowych w celu zresetowania ich wartości po każdym pomiarze.

Następnie pobierane są informacje o mierzonych wartościach z modułu INA219: napięcia bocznikowego – shuntvoltage, napięcia magistrali – busvoltage, prądu przepływającego w elemencie – current_mA oraz mocy – power_mW.

Kolejnym krokiem jest obliczenie wartości napięcia obciążenia, rezystancji oraz rezystywności.
Pomiary odbywają się co 5 sekund, a po każdym z nich zostają wyświetlone wszystkie zmierzone i obliczone wielkości na monitorze portu szeregowego.

W celu zminimalizowania błędu pomiaru został zaprojektowany algorytm liczący średnią wartość rezystywności, na podstawie której mikrokontroler decyduje jaki rodzaj metalu jest obecnie badany. Polega on na zapisaniu 5 ostatnich wartości rezystywności, które mieściły się w zbiorze od 0.001 Ohm • m do 0.1 Ohm • m, wszystkie inne zostają odrzucone. Po pięciu prawidłowych pomiarach obliczana jest średnia arytmetyczna i wyświetlana zostaje jej wartość.

Określenie rodzaju materiału z jakiego został stworzony element opiera się na sprawdzeniu wartości średniej rezystywności poprzez warunek „if”. Jeżeli wartość należy do danego zbioru, zostaje zmieniony stan z niskiego na wysoki w konkretnym wyjściu cyfrowym mikrokontrolera połączonego ze sterownikiem robota. Podliczana jest również liczba wszystkich posegregowanych elementów oraz ilość posegregowanych elementów danego materiału, która jest wyświetlana na monitorze portu szeregowego. Po tym procesie program zatrzymuje się na 10 sekund, tak aby robot miał czas na pobranie przedmiotu ze stacji pomiarowej.

Po upływie 10 sekund wartość średniej rezystywności jest sprowadzana do zera i następuje powrót do początku funkcji „obliczanie”.
