import numpy as np  # Import biblioteki NumPy pod aliasem np do operacji numerycznych.
import cv2  # Import biblioteki OpenCV do operacji na obrazach.
from PIL import Image  # Import klasy Image z modułu PIL do manipulacji obrazami.

# Funkcja do generowania macierzy DCT
def make_dct_matrix(n):
    i, j = np.meshgrid(np.arange(n), np.arange(n))  # Tworzy siatkę punktów dla macierzy indeksów.
    factor = np.sqrt(2 / n)  # Określa współczynnik dla transformacji DCT.
    dct = factor * np.cos((i + 0.5) * j * np.pi / n)  # Oblicza macierz DCT.
    dct[0] = np.sqrt(1 / n)  # Ustala pierwszy wiersz macierzy DCT.
    return dct  # Zwraca macierz DCT.

def process_image(with_noise=True):
    if with_noise:
        # Wczytuje obraz z szumem i normalizuje wartości pikseli.
        img_orig_with_noise = cv2.imread(r"").astype(np.float64) / 255
        size_2d_img = img_orig_with_noise.shape  # Określa wymiary obrazu.
    else:
        # Wczytuje obraz bez szumu i normalizuje wartości pikseli.
        img_orig = cv2.imread(r"").astype(np.float64) / 255
        size_2d_img = img_orig.shape  # Określa wymiary obrazu.

    size_img = size_2d_img[0] * size_2d_img[1] * size_2d_img[2]  # Określa liczbę pikseli.
    dct_matrix = make_dct_matrix(size_img)  # Generuje macierz DCT.

    percent = 0.06  # Procent pikseli branych pod uwagę podczas odbudowy.
    part = int(size_img * percent)  # Określa liczbę pikseli branych jako próbka.
    error_threshold = 0.01  # Określa próg błędu.

    i = 0  # Licznik iteracji.
    reconstructed = False  # Flaga odbudowy.
    max_iterations = size_img  # Maksymalna liczba iteracji.
    chosen_pixels = np.array([], dtype=int)  # Tablica indeksów wybranych pikseli.

    while not reconstructed and i < max_iterations:
        sample = np.random.choice(size_img, part, replace=True)  # Losuje próbkę pikseli.

        chosen_pixels = np.concatenate((chosen_pixels, sample))  # Łączy wybrane piksele.
        chosen_pixels = np.unique(chosen_pixels)  # Usuwa duplikaty indeksów pikseli.

        if with_noise:
            y = img_orig_with_noise.flat[chosen_pixels]  # Wybiera piksele z obrazu z szumem.
        else:
            y = img_orig.flat[chosen_pixels]  # Wybiera piksele z obrazu bez szumu.

        img = np.zeros(size_img)  # Inicjuje macierz zerową.
        img.flat[chosen_pixels] = y  # Ustawia wartości pikseli w macierzy.

        img_dct = dct_matrix.dot(img)  # Wykonuje transformację DCT.

        amplitude = 0.7  # Amplituda szumu impulsowego.
        noise_matrix = np.random.randn(size_img) * amplitude  # Tworzy macierz szumu.
        img_dct[chosen_pixels] += noise_matrix[chosen_pixels]  # Dodaje szum impulsowy do wybranych pikseli.

        img_idct = dct_matrix.T.dot(img_dct)  # Wykonuje odwrotną transformację DCT.
        img_idct_reshaped = img_idct.reshape(size_2d_img)  # Zmienia kształt macierzy do wymiarów obrazu.

        # Oblicza błąd pomiędzy oryginalnym obrazem a odbudowanym.
        error = np.mean((img_orig_with_noise if with_noise else img_orig - img_idct_reshaped) ** 2)

        # Reguluje próg błędu w zależności od rezultatu.
        if error < error_threshold:
            error_threshold /= 2
        else:
            part = min(part * 2, size_img)  # Zwiększa liczbę próbkowanych pikseli.

        # Tworzy maskę dla pikseli wybranych do odbudowy.
        img_dct_mask = np.zeros((size_2d_img[0], size_2d_img[1], size_2d_img[2]), dtype=np.uint8)
        img_dct_mask[:,:,0] = 101
        img_dct_mask[:,:,1] = 105
        img_dct_mask[:,:,2] = 97

        # Ustala piksele w masce na podstawie wybranych indeksów.
        for s in chosen_pixels:
            x = s % size_2d_img[1]
            y = s // size_2d_img[1]
            x_scaled = x // (size_2d_img[1] // size_2d_img[0])
            y_scaled = y // (size_2d_img[0] // size_2d_img[0])

            if x_scaled < size_2d_img[0] and y_scaled < size_2d_img[0]:
                img_dct_mask[y][x] = 255

        # Tworzy obraz z maską i zapisuje go jako plik PNG.
        img_dct_mask = Image.fromarray(img_dct_mask, mode='RGB')
        img_dct_mask.save(f'mask_{i}_{"with_noise" if with_noise else "without_noise"}.png')

        # Wczytuje obraz z maską.
        img_mask = cv2.imread(f'mask_{i}_{"with_noise" if with_noise else "without_noise"}.png')
        img_orig_masked = img_mask

        # Nakłada maskę na obraz oryginalny.
        img_orig_masked = np.where(img_mask == 255, (img_orig_with_noise if with_noise else img_orig) * 255, img_orig_masked)

        # Przygotowuje piksele dla odbudowanego obrazu.
        pixels = np.zeros((size_2d_img[0], size_2d_img[1], size_2d_img[2]), dtype=np.uint8)
        pixels[:,:,0] = 102
        pixels[:,:,1] = 189
        pixels[:,:,2] = 187

        # Wybiera piksele do odbudowy, zachowując te z oryginalnego obrazu.
        pixels = np.where(np.logical_and(img_orig_masked == (img_orig_with_noise if with_noise else img_orig) * 255, img_mask == 255), img_orig_masked, pixels)

        # Zapisuje odbudowany obraz jako plik PNG.
        cv2.imwrite(f'reconstructed_{i}_{"with_noise" if with_noise else "without_noise"}.png', pixels)

        # Wyświetla informację o numerze iteracji i liczbie pikseli w próbce.
        print(f"Iteration {i + 1}: Number of pixels in the sample: {len(chosen_pixels)}")

        # Wczytuje odbudowany obraz i sprawdza zgodność z oryginałem.
        reconstructed_image = cv2.imread(f'reconstructed_{i}_{"with_noise" if with_noise else "without_noise"}.png').astype(np.float64) / 255
        if np.allclose(img_orig_with_noise if with_noise else img_orig, reconstructed_image, rtol=0, atol=error_threshold):
            reconstructed = True  # Ustawia flagę odbudowy na True.
            break  # Przerywa pętlę.

        i += 1  # Zwiększa licznik iteracji.

# Wywołanie funkcji dla obrazu z szumem i bez szumu sekwencyjnie
process_image(True)  # Wywołanie funkcji dla obrazu z szumem.
process_image(False)  # Wywołanie funkcji dla obrazu bez szumu.
