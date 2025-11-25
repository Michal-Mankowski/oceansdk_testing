# Podobne do test2.py ale tym razem próbuje zrobić to jako ML, chociaż nie dokońca bo to nie zbyt generalizuje kwestia jest taka że wagi są dobierane na podstawie samych danych.
import dimod
import neal
from PIL import Image
import os

dataset = []
folder_path = "data"

for filename in sorted(os.listdir(folder_path)):
    
    file_path = os.path.join(folder_path, filename)
    img = Image.open(file_path).convert('1')
    pixels = [1 if p == 0 else -1 for p in list(img.getdata())]
    
    #Musimy ustawić jaką wartość funkcji do jakiej będziemy dążyć, patrząc na to że mamy 4 pixele, 4 i -4 uznałem za stosowne.
    #Chociaż i tak nie dobija do tych a bardziej do nich dąży... potem będziemy sprawdzać tylko czy jest większe czy mniejsze od zera
    #by widzieć czy jest bliżej tej -4 czy 4
    target = -4 if filename.startswith("I") else 4
    
    dataset.append({'pixels': pixels, 'target': target, 'name': filename})

#Tutaj robimy to samo co w test2.py tylko jako niewiadomo a nie manualne interakcje jak wtedy
pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
w = [dimod.Spin(f'w{u}{v}') for u,v in pairs]

bqm = dimod.BinaryQuadraticModel('SPIN')

for item in dataset:
    px = item['pixels']
    tgt = item['target']
    
    current_energy_expression = 0

    for idx, (u, v) in enumerate(pairs):
        weight_sign = px[u] * px[v]
        current_energy_expression += w[idx] * weight_sign
    
    diff = current_energy_expression - tgt
    squared_error = diff**2
    
    bqm += squared_error

sampler = neal.SimulatedAnnealingSampler()
answer = sampler.sample(bqm)

print(f"Best weights: {answer.first.sample}, Minimalny błąd: {answer.first.energy}")
