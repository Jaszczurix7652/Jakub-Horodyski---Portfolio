PL:

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


ENG:

2D block placement optimization involves finding the best arrangement of blocks in a given area to minimize unoccupied space. The BestFit algorithm used in the program implementation is a heuristic optimization algorithm that has been modified for this specific problem. In the code presented, the matplotlib library was used to generate charts and visualize the arrangement of blocks. The Rectangle class from the patches module in the matplotlib library was used to draw rectangles representing blocks.

At the beginning of the code there is a definition of the Block class, which represents a single block. Each block has a specific width, height, and x and y attributes that store the block's current coordinates on the grid.

The generate_blocks function generates a user-defined number of blocks with random dimensions. The width and height of the blocks are randomized from 1 to 5.

The main function pack_blocks performs block placement optimization. First, the blocks are sorted by area in descending order. A square grid is then created of an appropriate size that provides enough area to accommodate all the blocks.

In a loop that iterates over blocks, for each block all possible positions on the grid are checked. The algorithm finds a position that minimizes the unoccupied area on the grid. If such a position is found, the block (x and y) and grid coordinates are updated.
The draw_blocks function generates a graph visualizing the arrangement of blocks. The chart shows only those blocks that have x and y coordinates assigned. Blocks are represented as rectangles whose parameters are consistent with the block data (width, height, coordinates).

At the end of the code, the percentage of unoccupied space on the grid is calculated. The calculations are based on the total area of ​​the grid and the sum of the areas of the blocks that have assigned coordinates. Then the result is displayed on the screen.
Implementing the Python code allows you to understand how the 2D block placement optimization algorithm works. The use of the matplotlib library also allows you to visualize the results, which allows you to better understand the process of arranging blocks on the grid.

Sample result:

![Figure_1](https://github.com/user-attachments/assets/af0d23ae-267a-49cd-93ef-2901c012e457)
