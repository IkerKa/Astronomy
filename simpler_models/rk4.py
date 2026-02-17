import numpy as np
import matplotlib.pyplot as plt
import time
import os

### Constantes físicas

G = 6.67430e-11   # constante gravitacional
M = 1.989e30      # masa solar 
dt = 60 * 60 * 24 * 7  # 1 semana (paso más grande para ver problemas)
num_steps = 365 * 10000 // 7  

# Condiciones iniciales del planeta (similar a la Tierra) --> MODIFICAR

# Dependiendo de la orbita que queramos, tenemos que cambiar estas condiciones iniciales
pos = np.array([1.496e11, 0.0])     # posición inicial (m)  -> distancia promedio al sol
vel = np.array([0.0, 29.78e3])   # velocidad inicial (m/s) -> velocidad orbital promedio de la Tierra
masa = 5.972e24                     # masa del planeta (kg)

positions = []


def acceleration(position):
    r = -position # Vector unitario
    dist = np.linalg.norm(r) 
    rhat = r / dist

    return G * M / dist**2 * rhat
    
  
start_time = time.time()
for _ in range(num_steps):

    positions.append(pos.copy())

    #RK4
    k1_v = acceleration(pos)
    k1_r = vel

    k2_v = acceleration(pos + dt/2 * k1_r)
    k2_r = vel + dt/2 * k1_v

    k3_v = acceleration(pos + dt/2 * k2_r)
    k3_r = vel + dt/2 * k2_v

    k4_v = acceleration(pos + dt * k3_r)
    k4_r = vel + dt * k3_v

    pos += dt/6 * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    vel += dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)

elapsed_time = time.time() - start_time

# Ploteamos
positions = np.array(positions)

plt.figure(figsize=(6,6))
plt.plot(positions[:,0], positions[:,1], label='Órbita (RK4)')
plt.plot(0, 0, 'yo', label='Sol')
plt.axis('equal')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Órbita de la Tierra usando Runge-Kutta 4')
plt.legend()
plt.grid()
plt.show()

# Animación de la órbita
from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.5e11, 1.5e11)
ax.set_ylim(-1.5e11, 1.5e11)
ax.set_aspect('equal')
line, = ax.plot([], [], lw=2)
def init():
    line.set_data([], [])
    return line,
def update(frame):
    line.set_data(positions[:frame, 0], positions[:frame, 1])
    return line,

ax.plot(0, 0, 'yo', label='Sol')
ax.legend()
ani = FuncAnimation(fig, update, frames=len(positions), init_func=init, blit=True, interval=0.2)
plt.show()

plt.figure(figsize=(6,6))
plt.plot(positions[:365,0], positions[:365,1], 'b-', label='Órbita Euler 1 Año', alpha=0.5)
plt.plot(positions[:,0], positions[:,1], 'g-', label='Órbita Euler 300 Años', alpha=0.5)
plt.plot(0, 0, 'yo', label='Sol')
plt.axis('equal')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Comparación de Órbitas')
plt.legend()
plt.grid()
plt.show()

# guardar las posiciones en un archivo
# np.savetxt('orbita_rk4.txt', positions, header='x y', comments='')

anios = 10000
tiempo = elapsed_time
filename = 'tiempo_rk4.csv'
data = f"{anios},{tiempo:.10f}\n"

# Si el archivo no existe, escribe el encabezado
if not os.path.exists(filename):
    with open(filename, 'w') as f:
        f.write("Años,Tiempo (s)\n")

# Añade la nueva línea al final
with open(filename, 'a') as f:
    f.write(data)

# guardar animación como un archivo mp4
# ani.save('orbita_rk4.mp4', writer='ffmpeg', fps=30)
