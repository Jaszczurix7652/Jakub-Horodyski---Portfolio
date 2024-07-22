PL:

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

ENG:

The goal of this work is to analyze the feasibility of using an industrial arm robot to implement a selected industrial process related to the segregation of components
The scope of work included:
1. adaptation of an arm robot workstation to perform the selected industrial process;
2. development and implementation of a laboratory station;
3. development and execution of the robot control program using the RT-Toolbox3 environment.

In the repository was placed the program for measuring resistivity was written in the Arduino IDE environment in version 2.0.1.

At the beginning of the program, all the libraries used and their declaration along with the “namespace” were defined.

Next, the “setup” function was created, which includes initial settings for the serial port monitor, the INA219 module, defining the digital output ports, and displaying messages to the user whenever the program is turned on, such as “Hello!”, “Cannot detect sensor” - when the module is unconnected, and “Starting measurement...”. - when the module has been started correctly and the program can proceed to the next section.

In the next step, a “loop” was created, at the beginning of which all the values used for calculations as well as measurements were defined.

The next part of the program code is to set all the digital outputs of the controller to a low state, so that the bits will be reset on each new startup.

The next important part is to define and set the display of messages on the serial port monitor “Specify element diameter in meters: “ and “Specify element length in meters”. These messages are repeated every 10 seconds as a reminder. After the first message is displayed, the user must enter the appropriate dimension of the element, when he does so the loaded value will appear, followed immediately by a second message. Only after this is done will the program proceed to the next step.

The next part is the calculation of the cross-sectional area of the segregated element - S, based on the data previously entered by the user.

At this point, the program moves on to the most important part of the code which is the “calculate:” loop. At first, the low states on the digital outputs are set to reset their values after each measurement.

Next, information about the measured values is taken from the INA219 module: shuntvoltage - shuntvoltage, busvoltage - busvoltage, current flowing in the element - current_mA and power - power_mW.

The next step is to calculate the load voltage, resistance and resistivity values.
Measurements take place every 5 seconds, and after each measurement all measured and calculated values are displayed on the serial port monitor.

In order to minimize measurement error, an algorithm has been designed that counts the average value of resistivity, based on which the microcontroller decides what type of metal is currently being tested. It relies on storing the last 5 resistivity values that fell within the set from 0.001 Ohm - m to 0.1 Ohm - m, all others are discarded. After five valid measurements, the arithmetic average is calculated and its value is displayed.

Determination of the type of material from which the element was created is based on checking the value of the average resistivity through an “if” condition. If the value belongs to a given set, a state is changed from low to high in a specific digital output of the microcontroller connected to the robot controller. The number of all sorted items and the number of sorted items of a particular material are also counted and displayed on the serial port monitor. After this process, the program stops for 10 seconds so that the robot has time to retrieve the item from the measuring station.

After 10 seconds, the average resistivity value is brought to zero and the program returns to the beginning of the “calculation” function.
