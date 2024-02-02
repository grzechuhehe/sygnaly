import os
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from skimage import color, io, measure
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing, linear_model
from abc import abstractmethod
from typing import Optional
import pywt


def obcinanie_float(float_number, miejsca_po_przecinku):
    mnoznik = 10 ** miejsca_po_przecinku
    return int(float_number * mnoznik) / mnoznik


def adaptacyjny_wspolczynnik_kompresji(coeffs: object) -> object:
    if len(coeffs) % 2 != 0:
        del coeffs[-1]  # ostatni element nie jest istotny

    roznica_progowania = 0.0
    pozycja = 0

    for i in range(len(coeffs) - 1):
        roznica = coeffs[i + 1] - coeffs[i]
        if roznica > roznica_progowania:
            roznica_progowania = roznica
            pozycja = i

    wspolczynnik_kompresji = pozycja / len(coeffs)
    return obcinanie_float(wspolczynnik_kompresji, 2)


class TransformacjaKompresji:
    @abstractmethod
    def przod(self, zmienne: NDArray) -> NDArray:
        ...

    @abstractmethod
    def tyl(self, zmienne: NDArray) -> NDArray:
        ...


class TransformacjaFouriera2D(TransformacjaKompresji):
    def przod(self, zmienne: NDArray) -> NDArray:
        return np.fft.fft2(zmienne)

    def tyl(self, zmienne: NDArray) -> NDArray:
        return np.abs(np.fft.ifft2(zmienne))


class TransformacjaFalkowa2D(TransformacjaKompresji):
    def __init__(self, nazwa_falki: str, poziom: int):
        self.nazwa_falki = nazwa_falki
        self.poziom = poziom
        self.kawalki: Optional[NDArray] = None

    def przod(self, zmienne: NDArray) -> NDArray:
        przeksztalcone = pywt.wavedec2(zmienne, self.nazwa_falki, level=self.poziom)
        wspolczynniki, kawalki = pywt.coeffs_to_array(przeksztalcone)
        self.kawalki = kawalki
        return wspolczynniki

    def tyl(self, zmienne: NDArray) -> NDArray:
        if self.kawalki is None:
            raise ValueError(
                "Nie można wykonać odwrotnej transformacji bez wcześniejszego wykonania transformacji w przód!")
        zmienne = pywt.array_to_coeffs(zmienne, self.kawalki, output_format="wavedec2")
        return pywt.waverec2(zmienne, self.nazwa_falki)


def kompresja_i_dekompresja(zdjecie: NDArray, transformacja: TransformacjaKompresji, kompresja: float) -> NDArray:
    przeksztalcone = transformacja.przod(zdjecie)
    wspolczynniki = np.sort(np.abs(przeksztalcone.reshape(-1)))  # sortuj według magnitudy
    if kompresja < 0:
        kompresja = adaptacyjny_wspolczynnik_kompresji(wspolczynniki)

    progowanie = wspolczynniki[int(kompresja * len(wspolczynniki))]
    indeksy = np.abs(przeksztalcone) > progowanie

    zdekompresowane = przeksztalcone * indeksy
    return transformacja.tyl(zdekompresowane)


def zastosuj_rgb(funkcja: callable, zdjecie: NDArray, *args, **kwargs) -> NDArray:
    return np.dstack([funkcja(zdjecie[:, :, kanał], *args, **kwargs) for kanał in range(3)])


# kolorowe
zdjecie_kolorowe = io.imread(r"lemur.jpg")
io.imshow(zdjecie_kolorowe)
plt.title("Oryginalne zdjęcie kolorowe")
plt.show()

# szare
zdjecie_szare = color.rgb2gray(zdjecie_kolorowe)
io.imshow(zdjecie_szare, cmap="gray")
plt.title("Obraz szary")
plt.show()

# Rozmywanie gaussowskie kolorowego
rozmycie_gaussowskie = np.dstack([
    ndimage.gaussian_filter(zdjecie_kolorowe[:, :, kanał], 2)
    for kanał in range(3)
])
io.imshow(np.clip(rozmycie_gaussowskie.astype(int), 0, 255))
plt.title("Odszumianie - rozmycie gaussowskie")
plt.show()

# Transformacja Fouriera na kolorowym
fft_c_img = np.dstack([
    np.fft.fft2(zdjecie_kolorowe[:, :, kanał])
    for kanał in range(3)
])
io.imshow(np.clip(np.abs(fft_c_img).astype(int), 0, 255))
plt.title("Odszumianie - transformacja Fouriera")
plt.show()


# Filtrowanie obrazu za pomocą transformacji Fouriera
def filtrowanie_zdjecia(img, zachowana_frac=0.1):
    img_fft = img.copy()
    r = img_fft.shape[0]
    c = img_fft.shape[1]
    img_fft[int(r * zachowana_frac):int(r * (1 - zachowana_frac)), :] = 0
    img_fft[:, int(c * zachowana_frac):int(c * (1 - zachowana_frac))] = 0
    odfiltrowane_zdjecie = np.real(np.fft.ifft2(img_fft))
    return odfiltrowane_zdjecie


zdjecie_odszumione_fft = np.dstack([
    filtrowanie_zdjecia(fft_c_img[:, :, kanał])
    for kanał in range(3)
])
io.imshow(np.clip(zdjecie_odszumione_fft.astype(int), 0, 255))
plt.title("Odszumianie - transformacja Fouriera")
plt.show()

# Odszumianie falkowe szarego
falka = pywt.Wavelet("haar")
poziomy = int(np.floor(np.log2(zdjecie_szare.shape[0])))


def odszumianie_falkowe(zdjecie, falka, szum_sigma):
    wc = pywt.wavedec2(zdjecie_szare, falka, level=poziomy)
    arr, coeff_slices = pywt.coeffs_to_array(wc)
    arr = pywt.threshold(arr, szum_sigma, mode='soft')
    nwc = pywt.array_to_coeffs(arr, coeff_slices, output_format='wavedec2')
    return pywt.waverec2(nwc, falka)


zdjecie_szare_odszumione = odszumianie_falkowe(zdjecie_szare, falka, 0.0001)
io.imshow(zdjecie_szare_odszumione, cmap="gray")
plt.title("Odszumianie falkowe - szarego")
plt.show()

# Kompresja i dekompresja za pomocą transformacji Fouriera na zdjęciu kolorowym
zdjecie_zdekompresowane_fft = zastosuj_rgb(kompresja_i_dekompresja, zdjecie_kolorowe,
                                           transformacja=TransformacjaFouriera2D(), kompresja=-1)
io.imshow(np.clip(zdjecie_zdekompresowane_fft.astype(int), 0, 255))
plt.title("Kompresja - transformacja Fouriera")
plt.show()

# Kompresja i dekompresja za pomocą transformacji Fouriera na obrazie szarym
zdjecie_zdekompresowane_fft_szare = kompresja_i_dekompresja(zdjecie_szare, transformacja=TransformacjaFouriera2D(),
                                                            kompresja=0.45)
io.imshow(zdjecie_zdekompresowane_fft_szare, cmap="gray")
plt.title("Kompresja - transformacja Fouriera")
plt.show()

# Kompresja i dekompresja za pomocą transformacji falkowej na zdjęciu kolorowym
zdjecie_zdekompresowane_falkowe = zastosuj_rgb(kompresja_i_dekompresja, zdjecie_kolorowe,
                                               transformacja=TransformacjaFalkowa2D(nazwa_falki="haar", poziom=3),
                                               kompresja=0.96)
io.imshow(np.clip(zdjecie_zdekompresowane_falkowe.astype(int), 0, 255))
plt.title("Kompresja - transformacja falkowej")
plt.show()
