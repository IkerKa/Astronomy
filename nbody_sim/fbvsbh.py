import time
from body import Body
from simulator import Simulator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


#TODO: comparativa numérica de resultados

number = input("Barnes-Hut (1) o Fuerzas Brutas (2)? ")
if number == "1":
    print("Usando Barnes-Hut")
elif number == "2":
    print("Usando Fuerzas Brutas")

# Crear simulador
sim = Simulator(use_barnes_hut=(number == "1"), theta=0.5)  # Usar Barnes-Hut con theta más preciso


# Crear cuerpos
sol = Body("Sol ☀️", 1.989e30, [0, 0, 0], [0, 0, 0], 'yellow')
tierra = Body("Tierra 🌍", 5.972e24, [1.496e11, 0, 0], [0, 29.78e3, 0], 'blue')
marte = Body("Marte 🔴", 6.417e23, [2.279e11, 0, 0], [0, 24.077e3, 0], 'red')

# Añadir 1000 cuerpos aleatorios
for i in range(50):
    # if i % 100 == 0:
    #     # print(f"Añadiendo cuerpo {i + 1} de 1000")
    # Posición aleatoria en un rango de 0.5e11 a 3e11 metros desde el sol
    r = np.random.uniform(0.5e11, 3e11)
    theta = np.random.uniform(0, 2 * np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(-0.1e11, 0.1e11)
    # Velocidad tangencial aproximada para órbita circular
    v = np.sqrt(6.67430e-11 * sol.mass / r)
    vx = -v * np.sin(theta)
    vy = v * np.cos(theta)
    vz = np.random.uniform(-100, 100)
    mass = np.random.uniform(1e20, 1e22)
    color = 'gray'
    body = Body(f"Asteroide {i}", mass, [x, y, z], [vx, vy, vz], color)
    sim.add_body(body)

sim.add_body(sol)
sim.add_body(tierra)
sim.add_body(marte)

# Simular
dt = 60 * 60 * 24  # Un día en segundos
steps = 365 * 5

# Guardar trayectorias solo de Tierra, Sol y Marte
nombres_interes = ["Sol ☀️", "Tierra 🌍", "Marte 🔴"]
trayectorias = {body.name: [] for body in sim.bodies if body.name in nombres_interes}
time_start = time.time()

for i in range(steps):
    if i % 100 == 0:
        print(f"Paso {i + 1} de {steps} ({(i + 1) / steps * 100:.2f}%)", end='\r')
    sim.verlet_step(dt)
    for body in sim.bodies:
        if body.name in trayectorias:
            trayectorias[body.name].append(body.position.copy())

time_end = time.time()

plt.figure(figsize=(8, 8))
for name in nombres_interes:
    trayectoria = np.array(trayectorias[name])
    color = next(body.color for body in sim.bodies if body.name == name)
    if name == "Sol ☀️":
        plt.plot(trayectoria[:, 0], trayectoria[:, 1], 'o', label=name, color=color)
    else:
        plt.plot(trayectoria[:, 0], trayectoria[:, 1], label=name, color=color)

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Órbitas de Tierra, Sol y Marte")
plt.legend(loc='upper right', fontsize='small')
plt.axis('equal')
plt.grid()
plt.show()
