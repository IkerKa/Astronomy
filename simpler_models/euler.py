import numpy as np
import matplotlib.pyplot as plt
import time

### Constantes físicas

G = 6.67430e-11   # constante gravitacional
M = 1.989e30      # masa solar 
dt = 60 * 60 * 24 * 7
num_steps = 365 * 50 // 7
# Condiciones iniciales del planeta (similar a la Tierra) --> MODIFICAR

# Dependiendo de la orbita que queramos, tenemos que cambiar estas condiciones iniciales
pos = np.array([1.496e11, 0.0])     # posición inicial (m)  -> distancia promedio al sol
vel = np.array([0.0, 29.78e3])   # ← 29.78 km/s (velocidad orbital terrestre)     # velocidad inicial (m/s) -> velocidad orbital promedio de la Tierra
masa = 5.972e24                     # masa del planeta (kg)

positions = []

init_time = time.time()
for _ in range(num_steps):
    # Vector de posición al sol
    
    # F = G * ((m1m2) / r^2) * rhat

    # r es la distancia de los dos cuerpos, como el sol esta en el orígen
    r = -pos # Vector unitario
    dist = np.linalg.norm(r) 
    rhat = r / dist

    # La ley de gravitación universal incluye ambas masas, pero por ahora solo calcularemos la aceleracion, que no depende de la masa:
    # F = m * a --> a = F / m entonces si sustituimos a = G * ((M*m)/ r^2) * rhat) / m entonces m se cancela (siendo m la masa del planeta tierra)


    F = G * M / dist**2 * rhat
    
    acc = F # por lo comentado antes

    # Ahora que tenemos la informacion suficiente podemos aplicar la integración de euler (en el caso que estemos usando explicito, comentar estas dos proximas lineas)
    vel += acc * dt
    pos += vel * dt

    #Euler explícito -> descomentar si quiere evaluarse
    # pos += vel * dt
    # vel += acc * dt

    positions.append(pos.copy())


elapsed_time = time.time() - init_time



# Ploteamos
positions = np.array(positions)

plt.figure(figsize=(6,6))
plt.plot(positions[:,0], positions[:,1])
plt.plot(0, 0, 'yo', label='Sol')
plt.axis('equal')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Órbita usando método de Euler')
plt.legend()
plt.grid()
plt.show()

# Animación de la órbita
from matplotlib.animation import FuncAnimation
import os
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

#Comparar 1 año de orbita en euler vs 300
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

## TXT SAVING ##


# np.savetxt('orbita_euler.txt', positions, header='x y', comments='')


#guardar en un CSV el tiempo que ha tardado la simulación completa sin sobreescribir

# anios = 50        # Tiene que cuadrar con la cifra escrita al comienzo (paso temporal)
# tiempo = elapsed_time 
# filename = 'tiempo_euler.csv'
# data = f"{anios},{tiempo:.10f}\n"

# # Si el archivo no existe, escribe el encabezado
# if not os.path.exists(filename):
#     with open(filename, 'w') as f:
#         f.write("Años,Tiempo (s)\n")

# # Añade la nueva línea al final
# with open(filename, 'a') as f:
#     f.write(data)

# ani.save('orbita_euler.mp4', writer='ffmpeg', fps=60)


### Problema del método Euler

# Es inestable porque no conserva la energía. La orbita tiende a espiralar.


