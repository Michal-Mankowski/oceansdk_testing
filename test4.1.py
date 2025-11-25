#Test tego co w test4.py
from PIL import Image
import os

def model_test(pixels):
    px = [1 if p == 0 else -1 for p in pixels]
    b = [1, 1, 1, -1] #Przepisałem to co mi wyszło, ciekawostka mogą wychodzić inne bo lokalnie symulowane są troche losowe te wyniki
    w = [1, -1, 1, 1, -1, 1]
    
    energy = 0
    for i in range(4): energy += b[i] * px[i]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for idx, (u, v) in enumerate(pairs):
        energy += w[idx] * px[u] * px[v]
    #Jeżeli energia ujemna to wykryto I, jeśli nie no to nie
    return energy

folder_path = "data"
files = sorted(os.listdir(folder_path))

#Z jakiegoś powodu dla O12 wykrywa I, reszta git ale tak naprawdę test4.py to test możliwości i potencjału niż jakikolwiek poważny kod
#To wszystko i tak jest dla prezentacji i testów.
for filename in files:
    file_path = os.path.join(folder_path, filename)
    img = Image.open(file_path).convert('1')
    pixels = list(img.getdata())
    print(f"Dla pliku {filename} {"wykryto I" if model_test(pixels) < 0 else "nie wykryto I"}, a energia = {model_test(pixels)}")