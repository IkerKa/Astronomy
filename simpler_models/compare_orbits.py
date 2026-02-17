

### Leer archivos de trayectorias y compararlos en una animación
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Cargar las posiciones de las órbitas
euler_positions = np.loadtxt('orbita_euler.txt', skiprows=1)
rk4_positions = np.loadtxt('orbita_rk4.txt', skiprows=1)
verlet_positions = np.loadtxt('orbita_verlet.txt', skiprows=1)
leapfrog_positions = np.loadtxt('orbita_leapfrog.txt', skiprows=1)

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.5e11, 1.5e11)
ax.set_ylim(-1.5e11, 1.5e11)
ax.set_aspect('equal')
line_euler, = ax.plot([], [], lw=2, label='Euler')
line_rk4, = ax.plot([], [], lw=2, label='RK4')
line_verlet, = ax.plot([], [], lw=2, label='Verlet')
line_leapfrog, = ax.plot([], [], lw=2, label='Leapfrog')
head_euler, = ax.plot([], [], 'ro', markersize=6)  # Punto en la cabeza Euler
head_rk4, = ax.plot([], [], 'bo', markersize=6)    # Punto en la cabeza RK4
head_verlet, = ax.plot([], [], 'go', markersize=6)  # Punto en la cabeza Verlet
head_leapfrog, = ax.plot([], [], 'mo', markersize=6)  # Punto en la cabeza Leapfrog

def init():
    line_euler.set_data([], [])
    line_rk4.set_data([], [])
    line_verlet.set_data([], [])
    head_euler.set_data([], [])
    head_rk4.set_data([], [])
    head_verlet.set_data([], [])
    head_leapfrog.set_data([], [])
    return line_euler, line_rk4, line_verlet, line_leapfrog, head_euler, head_rk4, head_verlet, head_leapfrog

def update(frame):
    line_euler.set_data(euler_positions[:frame, 0], euler_positions[:frame, 1])
    line_rk4.set_data(rk4_positions[:frame, 0], rk4_positions[:frame, 1])
    line_verlet.set_data(verlet_positions[:frame, 0], verlet_positions[:frame, 1])
    head_euler.set_data([euler_positions[frame-1, 0]], [euler_positions[frame-1, 1]])
    head_rk4.set_data([rk4_positions[frame-1, 0]], [rk4_positions[frame-1, 1]])
    head_verlet.set_data([verlet_positions[frame-1, 0]], [verlet_positions[frame-1, 1]])
    head_leapfrog.set_data([leapfrog_positions[frame-1, 0]], [leapfrog_positions[frame-1, 1]])
    return line_euler, line_rk4, line_verlet, line_leapfrog, head_euler, head_rk4, head_verlet, head_leapfrog

ax.plot(0, 0, 'yo', label='Sol')
ax.legend()

ani = FuncAnimation(fig, update, frames=len(euler_positions), init_func=init, blit=True, interval=0.2)
plt.show()

# guardar la animación como un archivo mp4
# ani.save('comparacion_orbitas.mp4', writer='ffmpeg', fps=30)
