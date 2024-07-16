Celem projektu była analiza pracy algorytmów decyzyjnych na podstawie symulacji przebiegu licytacji. Badanym algorytmem był minmax, którego zadaniem jest znalezienie najlepszej oferty, jaką gracz może złożyć, biorąc pod uwagę jego bieżący budżet i cechy charakteru. 

By móc zrealizować projekt został stworzony program pozwalający na zasymulowanie procesu licytacji z wykorzystaniem algorytmu minmax. Może w niej uczestniczyć dowolna liczba graczy, każdemu z nich przypisuje się dostępny kapitał oraz wartości cech, które wpływają na chęć podbicia stawki. Program składa się z dwóch klas: Player i Auction. Podczas inicjacji aukcji podawana jest nazwa przedmiotu, który ma zostać poddany licytacji oraz jego cena wywoławcza. Aukcja jest podzielona na rundy, w każdej z nich każdy z graczy określa o ile podbije kwotę licytacji, w przypadku pierwszej rundy jest to cena wywoławcza, w przypadku kolejnych jest to najwyższa kwota zakupy zaproponowana w rundzie wcześniejszej. Gracz poddaje się w momencie braków środków umożliwiających podbicie kwoty licytacji. By móc przeprowadzić odpowiednie badania możliwe jest określenie ile aukcji ma zostać zasymulowane, następnie są wyświetlane wykresy przedstawiające jaką ofertę za licytowany przedmiot złożył każdy z graczy podczas każdej z rund aukcji. Po czym na podstawie wyników wszystkich przeprowadzonych aukcji wyświetlany jest wykres średniej wartości stawek każdego z graczy. Gdy przykładowo zasymulowano dziesięć aukcji i podczas z nich gracz dziewięć razy ukończył licytacje po czterech rundach, a raz po trzech to do wykresu średniej wartości stawek zostaną wzięte wartości stawek tylko z trzech rund, każdej z dziesięciu aukcji.

Program został podzielony na poszczególne klasy, pierwsza z nich „Player” pozwala na zdefiniowanie każdego z graczy. Przechowuje informacje o graczu oraz określa o ile chce podbić kwotę licytacji, uwzględniając przy tym cechy jego osobowości.

Dla każdego gracza wykonywane są następujące funkcje:

•	bid – funkcja odpowiedzialna za decyzję gracza o złożeniu oferty. Gracz podejmuje decyzję, czy podnieść stawkę i o jaką kwotę. Wykorzystuje informacje o aktualnych ofertach innych graczy, poprzednim wzroście wielkości oraz pozostałych graczach.

•	minmax – działa w ramach ograniczonej przestrzeni stanów, analizując różne możliwe wartości podbicia stawki (do maksymalnie 10 jednostek lub dostępnego budżetu). Dla każdej możliwej oferty oblicza użyteczność na podstawie cech gracza. Następnie wybiera ofertę, która maksymalizuje użyteczność, i aktualizuje budżet gracza.

•	utility – funkcja obliczająca wartości użyteczności danej oferty, uwzględniająca cechy gracza, wysokość podbicia ceny przedmiotu oraz informacji o pozostałych graczach.

Druga klasa „Auction” ma za zadanie odpowiednie przeprowadzenie i zakończenie licytacji. Posiada ona informacje o licytowanym przedmiocie i śledzi przebieg aukcji pilnując o ile zwiększyła się kwota aukcji podczas każdej z rund. Klasa ta określa kiedy aukcja ma się zakończyć i ogłasza zwycięzcę.

Dla każdej aukcji wykonywane są następujące funkcje:

•	play – funkcja, która przeprowadza aukcję. Gracze składają oferty w rundach, aż zostanie tylko jeden gracz zdolny do dalszego podbijania stawki lub wszyscy gracze spasują. Metoda wyświetla wyniki każdej rundy oraz ogłasza zwycięzcę.

•	plot_bid_history – funkcja ta generuje wykres przebiegu aukcji pokazując jaką kwotę licytacji złożył każdy z graczy podczas każdej z rund 

Poza klasami znajduje się funkcja count_avg_bid, służy do obliczenia średnich wartości ofert składanych przez graczy w serii aukcji. Najpierw wyświetla historię ofert każdego gracza dla każdej aukcji. Następnie oblicza sumaryczne oferty graczy, uwzględniając każdą aukcję. Poprawia sumaryczne oferty, aby zapewnić, że długości historii ofert dla poszczególnych graczy są zgodne. Na koniec oblicza średnie wartości ofert dla każdego gracza.