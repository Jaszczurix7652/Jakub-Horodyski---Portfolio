# importujemy bibliotekę do tworzenia wykresów
import matplotlib.pyplot as plt
# importujemy klasę Rectangle z biblioteki matplotlib.patches
from matplotlib.patches import Rectangle
# importujemy moduł random
import random

# definujemy klasę Block
class Block:
    # konstruktor klasy Block
    def __init__(self):
        # losujemy wartość szerokości blocka z zakresu od 1 do 5
        self.width = random.randint(1, 5)
        # losujemy wartość wysokości blocka z zakresu od 1 do 5
        self.height = random.randint(1, 5)
        # ustawiamy wartości x i y na None
        self.x = None
        self.y = None

# funkcja pack_blocks
def pack_blocks(blocks):
    # sortujemy listę blocków w kolejności malejącej wartości pola powierzchni każdego blocka
    blocks = sorted(blocks, key=lambda b: b.width * b.height, reverse=True)
    # obliczamy całkowitą powierzchnię blocków
    total_area = sum(b.width * b.height for b in blocks)
    # obliczamy długość boku kwadratu, który pozwoli na umieszczenie wszystkich blocków
    square_size = int(total_area ** 0.5) + 1
    # tworzymy siatkę z wartościami None o wymiarach square_size x square_size
    grid = [[None for _ in range(square_size)] for _ in range(square_size)]
    # dla każdego blocka w liście blocks
    for block in blocks:
        # iterujemy po każdym polu siatki, w którym można umieścić blocka
        for y in range(square_size - block.height + 1):
            for x in range(square_size - block.width + 1):
                # sprawdzamy, czy wszystkie pola siatki, na których ma się znaleźć block, są puste
                if all(grid[y + j][x + i] is None for i in range(block.width) for j in range(block.height)):
                    # jeśli tak, to umieszczamy blocka w siatce
                    block.x, block.y = x, y
                    for j in range(block.height):
                        for i in range(block.width):
                            grid[y + j][x + i] = block
                    break
            # jeśli udało się umieścić blocka, to przerywamy pętlę
            if block.x is not None:
                break
    # zwracamy siatkę
    return grid

# funkcja draw_blocks
def draw_blocks(blocks):
    # Tworzymy nowy rysunek i osie
    fig, ax = plt.subplots()
    # Ustawiamy proporcje osi rysunku na równą, tak aby rysunek nie był zdeformowany
    ax.set_aspect('equal')
    # Tworzymy listę ważnych blocków, czyli tych, które udało się umieścić w siatce
    valid_blocks = [b for b in blocks if b.x is not None and b.y is not None]
    # Ustawiamy ograniczenia osi x i y na podstawie pozycji i wymiarów ważnych blocków
    ax.set_xlim([0, max(b.x + b.width for b in valid_blocks)])
    ax.set_ylim([0, max(b.y + b.height for b in valid_blocks)])
    # Dla każdego ważnego blocka dodajemy do rysunku patch w kształcie prostokąta o określonej pozycji i wymiarach
    for block in valid_blocks:
        ax.add_patch(Rectangle((block.x, block.y), block.width, block.height, edgecolor='black', facecolor='grey'))
    # Wyświetlamy rysunek
    plt.show()

# Tworzymy listę 20 losowych blocków
blocks = [Block() for _ in range(50)]
# Umieszczamy blocki w siatce
grid = pack_blocks(blocks)
# Rysujemy wykres z umieszczonymi blockami
draw_blocks(blocks)