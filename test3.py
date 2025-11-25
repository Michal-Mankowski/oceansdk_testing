#Tesotwanie funkcji RelU używając dwave optimisation, ogólnie z tego co rozumiem to używa hybrydy a nie w pełni kwantowo
import dwave.optimization
import numpy as np

model = dwave.optimization.Model()

x = model.integer(1, lower_bound=-10, upper_bound=10)

objective_function = x**2 - 16 #Przykładowa funkcja celu

zero = model.constant(0)

relu_output = dwave.optimization.maximum(objective_function, zero) #Relu przepuszcza tylko wartości wieksze niż zero, reszte zeruje

model.minimize(relu_output)

#Nie mogę odpalić tego minimalizacji bo to wymaga usługi w chmurze ale można jakby ręcznie testować wartości

model.states.resize(1)
with model.lock():
    best_x = None
    min_energy = float('inf')

    for test_val in range(-10, 11):
        x.set_state(0, np.array([test_val]))
        current_energy = model.objective.state(0).item()

        if current_energy < min_energy:
            min_energy = current_energy
            best_x = test_val
            
        print(f"x = {test_val:<5} | f(x) = {current_energy}")

print(f"Minimum: x={best_x}, energia={min_energy}")
#Starałem się dobrać funkcje celu tak by było widać działanie RelU