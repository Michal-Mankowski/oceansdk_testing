# Test dla najprostszej funkcji celu
import dimod
import neal

sampler = neal.SimulatedAnnealingSampler() #Symulowany sampler

x = dimod.Binary('x') #Tworzymy kubit

objective_function = x**2 + 1 # Tworzymy porstą funkcje celu 

answer = sampler.sample(objective_function) #Funkcje celu dajemy do samplera

#Wyświetlamy wyniki
print(f"Wartość zmiennej: {answer.first.sample['x']} Energia: {answer.first.energy}")


