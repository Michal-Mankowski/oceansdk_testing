#Bardziej skomplikowany przykład
#Zdecydowałem się wygenerować zdjęcia 2x2 biało czarne i rozpoznawać gdzie pojawia się litera I
#Czyli 2 pixele czarne pionowo i 2 białe obok (niezaleznie od strony) 
#Pliki których nazwa się zaczyna na I zawierają litere I a te co się zaczynają na O nie ma.
import dimod
import neal
from PIL import Image
import os

sampler = neal.SimulatedAnnealingSampler() 

bqm = dimod.BinaryQuadraticModel('SPIN') #Używam spin bo on ma -1 i 1 zamiast 0 i 1 więc białe piksele mają większe znaczenie.
#Potencjalnie spina może będziemy używać bo chcemy wagi w 1 i -1? ewentualnie będziemy kodować wagi dwoma bitami jeśli chcemy 3 wartości

#Patrzymy na obrazek tak:
#q0  q1
#q2 q3

#Jeśli interackkcja ma wartość -1, to znaczy że chcemy by były takie same (no bo szukamy minimum)
bqm.add_interaction('q0', 'q2', -1)
bqm.add_interaction('q1', 'q3', -1)

#Tutaj chcemy by były różne, więc interakcja +1
bqm.add_interaction('q0', 'q1', 1)
bqm.add_interaction('q2', 'q3', 1)

bqm.add_interaction('q0', 'q3', 1)
bqm.add_interaction('q1', 'q2', 1)


answer = sampler.sample(bqm)
lowest_energy = answer.first.energy

folder_path = "data"
files = sorted(os.listdir(folder_path))

for filename in files:
    file_path = os.path.join(folder_path, filename)
    img = Image.open(file_path).convert('1')
    pixels = list(img.getdata())
    #Kodowanie obrazu
    sample = {
        'q0': 1 if pixels[0] == 0 else -1,
        'q1': 1 if pixels[1] == 0 else -1,
        'q2': 1 if pixels[2] == 0 else -1,
        'q3': 1 if pixels[3] == 0 else -1,
    }
    answer = sampler.sample(bqm)
    lowest_energy = answer.first.energy

    energy = bqm.energy(sample)

    #Sprawdzamy czy energia układu dla wartości zapodanych dla obrazka ma najmniejszą możliwą energie czyli poprawne rozwiazanie.
    if (energy == lowest_energy):
        print(f"dla pliku: {filename} znaleziono I")
    else:
        print(f"dla pliku: {filename} nie znaleziono I")

terms = []
for (var1, var2), weight in bqm.quadratic.items():
    terms.append(f"({weight})*{var1}*{var2}")
equation = " + ".join(terms)
print(f"E = {equation}") # Równanie z wagami, które powstało z tych add_interaction
