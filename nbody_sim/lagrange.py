import time
import matplotlib.pyplot as plt
import numpy as np
from body import Body
from simulator import Simulator

G = 6.67430e-11  

sim = Simulator(use_barnes_hut=False)

m = 5.972e24        # masa Tierra
R = 4e8             # radio al centro de masa (distancia de cada masa al centro)
dt = 60 * 60 * 6    # 6 horas
steps = 365 * 2     # 2 años simulados

# Velocidad orbital circular correcta para triángulo equilátero
v_circular = np.sqrt(G * m / (np.sqrt(3) * R))

# Posiciones iniciales (triángulo equilátero)
posiciones = [
    [R, 0, 0],
    [-R/2,  (np.sqrt(3)/2)*R, 0],
    [-R/2, -(np.sqrt(3)/2)*R, 0]
]

# Velocidades iniciales (perpendiculares al radio, sentido antihorario)
velocidades = []
for pos in posiciones:
    r_vec = np.array(pos)
    r_norm = np.linalg.norm(r_vec)
    tang = np.array([-r_vec[1], r_vec[0], 0]) / r_norm
    vel = tang * v_circular
    velocidades.append(list(vel))

colores = ['blue', 'red', 'green']
nombres = ['Cuerpo A', 'Cuerpo B', 'Cuerpo C']

bodies = []
for nombre, pos, vel, col in zip(nombres, posiciones, velocidades, colores):
    body = Body(nombre, m, pos, vel, col)
    sim.add_body(body)
    bodies.append(body)

trajectory_bodies = {body.name: [] for body in bodies}

time_start = time.time()

for step in range(steps):
    sim.leapfrog_step(dt)
    for body in bodies:
        trajectory_bodies[body.name].append(body.position.copy())

time_end = time.time()

trajectories = {name: np.array(pos) for name, pos in trajectory_bodies.items()}

plt.figure(figsize=(6, 6))
for name, traj in trajectories.items():
    plt.plot(traj[:, 0] / 1e3, traj[:, 1] / 1e3, label=name, linewidth=2)
# Unir los cuerpos con líneas rectas en cada frame final
last_positions = [traj[-1] for traj in trajectories.values()]
for i in range(len(last_positions)):
    p1 = last_positions[i]
    p2 = last_positions[(i + 1) % len(last_positions)]
    plt.plot([p1[0] / 1e3, p2[0] / 1e3], [p1[1] / 1e3, p2[1] / 1e3], 'k-.', linewidth=1.5)
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.title("Configuración de Lagrange (Triángulo equilátero)")
plt.legend()
plt.axis('equal')
plt.grid(True)

from matplotlib.animation import FuncAnimation

def update(frame):
    plt.clf()
    current_positions = []
    for name, traj in trajectories.items():
        plt.plot(traj[:frame, 0] / 1e3, traj[:frame, 1] / 1e3, label=name, linewidth=2)
        plt.scatter(traj[frame-1, 0] / 1e3, traj[frame-1, 1] / 1e3, s=80)
        current_positions.append(traj[frame-1])
    # Unir los cuerpos con líneas rectas en cada frame
    for i in range(len(current_positions)):
        p1 = current_positions[i]
        p2 = current_positions[(i + 1) % len(current_positions)]
        plt.plot([p1[0] / 1e3, p2[0] / 1e3], [p1[1] / 1e3, p2[1] / 1e3], 'k-.', linewidth=1.5)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title("Configuración de Lagrange (Triángulo equilátero)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

ani = FuncAnimation(plt.gcf(), update, frames=len(trajectories['Cuerpo A']), interval=50)
plt.show()
plt.close()
