PL:

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

ENG:

The goal of the project was to analyze the performance of decision-making algorithms based on simulations of the bidding process. The algorithm studied was minmax, whose task is to find the best bid a player can make, taking into account his current budget and character traits. 

To be able to carry out the project, a program was created to simulate the bidding process using the minmax algorithm. Any number of players can participate in it, each of them is assigned available capital and values of traits that affect the willingness to raise the stake. The program consists of two classes: Player and Auction. During the initiation of an auction, the name of the item to be auctioned and its starting price are given. The auction is divided into rounds, in each of which each player determines by how much he will raise the bidding amount, in the case of the first round it is the starting price, in the case of subsequent rounds it is the highest purchase amount offered in the earlier round. A player surrenders when there is a shortage of funds to raise the bidding amount. In order to be able to carry out the relevant research, it is possible to determine how many auctions are to be simulated, then charts are displayed showing what bid for the auctioned item each player made during each round of the auction. After which, a graph of the average value of each player's stakes is displayed based on the results of all the auctions held. When, for example, ten auctions were simulated and during one of them a player finished bidding after four rounds nine times, and once after three, then for the graph of the average value of the stakes will be taken the values of the stakes of only three rounds, each of the ten auctions.

The program is divided into different classes, the first of which “Player” allows you to define each player. It stores information about the player and determines by how much he wants to raise the bidding amount, taking into account his personality traits.

The following functions are performed for each player:

• bid - the function responsible for the player's decision to bid. The player decides whether to raise the bid and by what amount. It uses information about the current bids of other players, the previous increase in size and other players.

• minmax - works within a limited state space, analyzing various possible values of a raise (up to a maximum of 10 units or the available budget). For each possible bid, it calculates utility based on player characteristics. It then selects the offer that maximizes utility and updates the player's budget.

• utility - a function that calculates the utility values of a given bid, taking into account the player's characteristics, the amount of the item's price raise and information about the other players.

The second class “Auction” has the task of properly conducting and completing the bidding. It has information about the item being auctioned and tracks the course of the auction keeping an eye on how much the auction amount has increased during each round. This class determines when the auction should end and announces the winner.

The following functions are performed for each auction:

• play - the function that conducts the auction. Players bid in rounds until there is only one player left capable of raising the bid further or all players fold. The method displays the results of each round and announces the winner.

• plot_bid_history - this function generates a graph of the auction progress showing what amount of bidding each player placed during each round. 

Outside the classes is the count_avg_bid function, used to calculate the average bids placed by players in a series of auctions. It first displays each player's bid history for each auction. Then it calculates the aggregate bids of players, taking into account each auction. It corrects the aggregate bids to ensure that the bid history lengths for each player match. Finally, it calculates the average bid values for each player.
