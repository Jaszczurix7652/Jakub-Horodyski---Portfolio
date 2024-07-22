Optymalizacja rozmieszczenia bloków 2D polega na znalezieniu najlepszego układu bloków na danym obszarze w celu minimalizacji niezajętej powierzchni. Algorytm BestFit, użyty w implementacji programu, jest heurystycznym algorytmem optymalizacyjnym, który został zmodyfikowany dla tego konkretnego problemu. W przedstawionym kodzie, wykorzystano bibliotekę matplotlib do generowania wykresów i wizualizacji rozmieszczenia bloków. Do rysowania prostokątów reprezentujących bloki użyto klasy Rectangle z modułu patches w bibliotece matplotlib.

Na początku kodu znajduje się definicja klasy Block, która reprezentuje pojedynczy blok. Każdy blok ma określoną szerokość, wysokość oraz atrybuty x i y, które przechowują aktualne współrzędne bloku na siatce.

Funkcja generate_blocks generuje określoną liczbę bloków przez użytkownika o losowych wymiarach. Szerokość i wysokość bloków są losowane z zakresu od 1 do 5.

Główna funkcja pack_blocks realizuje optymalizację rozmieszczenia bloków. Na początku bloki są sortowane według pola powierzchni w kolejności malejącej. Następnie tworzona jest kwadratowa siatka o odpowiednim rozmiarze, który zapewnia wystarczającą powierzchnię do umieszczenia wszystkich bloków.

W pętli iterującej po blokach, dla każdego bloku sprawdzane są wszystkie możliwe pozycje na siatce. Algorytm znajduje pozycję, która minimalizuje niezajętą powierzchnię na siatce. Jeśli taka pozycja zostanie znaleziona, to aktualizowane są współrzędne bloku (x i y) oraz siatki.
Funkcja draw_blocks generuje wykres z wizualizacją rozmieszczenia bloków. Na wykresie przedstawiane są tylko te bloki, które mają przypisane współrzędne x i y. Bloki są reprezentowane jako prostokąty, których parametry są zgodne z danymi bloków (szerokość, wysokość, współrzędne).

Na końcu kodu obliczany jest procentowy udział niezajętej powierzchni na siatce. Obliczenia są oparte na całkowitej powierzchni siatki oraz sumie powierzchni bloków, które mają przypisane współrzędne. Następnie wynik jest wyświetlany na ekranie.
Implementacja kodu w języku Python pozwala na zrozumienie sposobu działania algorytmu optymalizacji rozmieszczenia bloków 2D. Wykorzystanie biblioteki matplotlib umożliwia również wizualizację wyników, co pozwala lepiej zrozumieć proces rozmieszczania bloków na siatce.

Przykładowy wynik:

![Figure_1](https://github.com/user-attachments/assets/af0d23ae-267a-49cd-93ef-2901c012e457)
