# main.py
from body import Body
from simulator import Simulator

# Crear simulador
sim = Simulator()

sol = Body("Sol", 1.989e30, [0, 0], [0, 0])
tierra = Body("Tierra", 5.972e24, [1.496e11, 0], [0, 29.78e3])

# La Tierra está a 1 UA del Sol y se mueve a 29.78 km/s en dirección perpendicular al Sol

# Añadir cuerpos
sim.add_body(sol)
sim.add_body(tierra)

# Calcular aceleración que siente la Tierra
acc_tierra = sim.compute_acceleration(tierra)

print("Aceleración sobre la Tierra:", acc_tierra)

# verificacion de la aceleración
# a_centripeta = v**2 / r donde r = sqrt(x^2 + y^2)
acc_tierra_verificacion = tierra.velocity[1]**2 / (tierra.position[0]**2 + tierra.position[1]**2)**0.5
print("Aceleración centrípeta esperada:", acc_tierra_verificacion)

# Resultado: [-0.00593167  0.        ]
# Componente X negativa significa que la Tierra es atraída hacia el Sol, y la componente Y es cero porque la Tierra se mueve en un plano perpendicular al Sol.
# entonces, al ser la velocidad tangencial, la aceleración es centrípeta por lo que asi se forma un movimiento circular.