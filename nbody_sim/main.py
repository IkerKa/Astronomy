import time
from body import Body
from simulator import Simulator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

G = 6.67430e-11  # Constante gravitacional

def add_random_orbiting_body(center_body, name_prefix="Asteroide", mass_range=(1e20, 1e22),
                              r_range=(0.5e11, 3e11), z_range=(-0.05e11, 0.05e11),
                              eccentricity=0.0, color='gray'):
    """
    Crea un cuerpo con condiciones iniciales estables para orbitar alrededor de `center_body`.
    """
    # Distancia radial en el plano XY
    r = np.random.uniform(*r_range)
    theta = np.random.uniform(0, 2 * np.pi)

    # Posición inicial
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(*z_range)

    # Velocidad orbital circular
    v_circular = np.sqrt(G * center_body.mass / r)

    # Ajuste por excentricidad
    v = v_circular * np.sqrt((1 - eccentricity) / (1 + eccentricity))

    # Dirección tangencial (perpendicular al radio)
    vx = -v * np.sin(theta)
    vy = v * np.cos(theta)
    vz = 0.0

    # Masa aleatoria
    mass = np.random.uniform(*mass_range)

    # Crear y devolver el cuerpo
    return Body(f"{name_prefix}", mass, [x, y, z], [vx, vy, vz], color)


# Crear simulador
sim = Simulator(use_barnes_hut=True, theta=0.3)  

## Sistema solar ##
sol = Body("Sol", 1.989e30, [0, 0, 0], [0, 0, 0], color='yellow')

tierra_pos = [1.496e11, 0, 0]  # Posición inicial (m)
tierra_vel = [0, 30.29e3, 0]   # Velocidad inicial (m/s)
tierra = Body("Tierra", 5.972e24, tierra_pos, tierra_vel, color='#6B93D6')

# Mercurio
mercurio_pos = [4.600e10, 0, 0]  
mercurio_vel = [0, 58.98e3, 0]   
mercurio = Body("Mercurio", 3.301e23, mercurio_pos, mercurio_vel, color='#8C7853')

# Venus
venus_pos = [1.075e11, 0, 0]     
venus_vel = [0, 35.26e3, 0]     
venus = Body("Venus", 4.867e24, venus_pos, venus_vel, color='#FFC649')


# Marte
marte_pos = [2.067e11, 0, 0]    
marte_vel = [0, 26.50e3, 0]      
marte = Body("Marte", 6.417e23, marte_pos, marte_vel, color='#CD5C5C')

# # Júpiter
jupiter_pos = [7.405e11, 0, 0]   
jupiter_vel = [0, 13.72e3, 0]    
jupiter = Body("Júpiter", 1.898e27, jupiter_pos, jupiter_vel, color='#D8CA9D')

# Saturno
saturno_pos = [1.352e12, 0, 0]   
saturno_vel = [0, 10.18e3, 0]    
saturno = Body("Saturno", 5.683e26, saturno_pos, saturno_vel, color='#FAD5A5')

# Urano
urano_pos = [2.741e12, 0, 0]     
urano_vel = [0, 7.11e3, 0]       
urano = Body("Urano", 8.681e25, urano_pos, urano_vel, color='#4FD0E7')

# Neptuno
neptuno_pos = [4.444e12, 0, 0]   
neptuno_vel = [0, 5.50e3, 0]     
neptuno = Body("Neptuno", 1.024e26, neptuno_pos, neptuno_vel, color='#4B70DD')

bodies = [sol, tierra, marte, venus, mercurio, jupiter, saturno, urano, neptuno]

sim.add_body(sol)
sim.add_body(tierra)
sim.add_body(mercurio)
sim.add_body(venus)
sim.add_body(marte)
sim.add_body(jupiter)
# sim.add_body(saturno)
# sim.add_body(urano)
# sim.add_body(neptuno)

# Añadir asteroides en órbita estable alrededor del Sol
for i in range(10):
    asteroid = add_random_orbiting_body(
        sol,
        name_prefix=f"Asteroide {i+1}",
        eccentricity=np.random.uniform(0.05, 0.1),  # Menor excentricidad para órbitas más estables
        color='gray'
    )
    # Ajustar posición y velocidad relativas al Sol
    asteroid.position = np.array(asteroid.position) + np.array(sol.position)
    asteroid.velocity = np.array(asteroid.velocity) + np.array(sol.velocity)
    bodies.append(asteroid)
    sim.add_body(asteroid)



# Añadir cuerpos al simulador
# for body in bodies:
#     sim.add_body(body)

print(f"🌌 Simulando {len(bodies)} cuerpos celestes...")


# dt = 60 * 60 * 24 * 7  # Una semana
#1 hora
dt = 60 * 60 * 24 * 7  # Una semana
steps = 365 * 2        # 2 años

trajectory_bodies = {body.name: [] for body in bodies}
time_points = []

time_start = time.time()

for step in range(steps):
    sim.leapfrog_step(dt)
    
    for body in bodies:
        trajectory_bodies[body.name].append(body.position.copy())
    
    time_points.append(step * dt / (365.25 * 24 * 3600))
    
    if step % (365 // 7) == 0:
        years = step * dt / (365.25 * 24 * 3600)
        print(f"Año {years:.1f} simulado")

time_end = time.time()

trajectories = {name: np.array(pos) for name, pos in trajectory_bodies.items()}


# Gráfico 2D combinado: Órbitas, distancia y velocidad
fig, axs = plt.subplots(3, 1, figsize=(10, 14))

# Plot 1: Órbitas elípticas (2D)
for name, traj in trajectories.items():
    if name != "Sol":
        axs[0].plot(traj[:, 0]/1e11, traj[:, 1]/1e11, label=name, linewidth=2)
axs[0].plot(0, 0, 'yo', markersize=10, label='Sol')
axs[0].set_xlabel("x (UA)")
axs[0].set_ylabel("y (UA)")
axs[0].set_title("Órbitas Elípticas - Vista General")
axs[0].legend()
axs[0].axis('equal')
axs[0].grid(True, alpha=0.3)

# Plot 2: Distancia al Sol vs tiempo
for name, traj in trajectories.items():
    if name != "Sol":
        distances = np.linalg.norm(traj, axis=1) / 1e11
        axs[1].plot(time_points, distances, label=f'Distancia {name}', linewidth=2)
axs[1].set_xlabel("Tiempo (años)")
axs[1].set_ylabel("Distancia al Sol (UA)")
axs[1].set_title("Variación de Distancia")
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# Plot 3: Velocidad vs tiempo
for name, traj in trajectories.items():
    if name != "Sol":
        velocities = []
        for i in range(1, len(traj)):
            v = np.linalg.norm(traj[i] - traj[i-1]) / dt / 1000  # km/s
            velocities.append(v)
        axs[2].plot(time_points[1:], velocities, label=f'Velocidad {name}', linewidth=2)
axs[2].set_xlabel("Tiempo (años)")
axs[2].set_ylabel("Velocidad (km/s)")
axs[2].set_title("Variación de Velocidad")
axs[2].legend()
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

fig3d = plt.figure(figsize=(8, 8))
ax3d = fig3d.add_subplot(111, projection='3d')
for name, traj in trajectories.items():
    if name != "Sol":
        ax3d.plot3D(traj[:, 0]/1e11, traj[:, 1]/1e11, traj[:, 2]/1e11, label=name, linewidth=2)
ax3d.scatter(0, 0, 0, color='yellow', s=100, label='Sol')
ax3d.set_xlabel("x (UA)")
ax3d.set_ylabel("y (UA)")
ax3d.set_zlabel("z (UA)")
ax3d.set_title("Órbitas Elípticas - Vista 3D")
ax3d.legend()
ax3d.grid(True, alpha=0.3)
plt.show()

def set_axes_equal(ax):
    '''Set equal scale for 3D plot axes.'''
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    spans = limits[:,1] - limits[:,0]
    centers = np.mean(limits, axis=1)
    max_span = max(spans)
    new_limits = np.array([
        centers - max_span/2,
        centers + max_span/2
    ]).T
    ax.set_xlim3d(new_limits[0])
    ax.set_ylim3d(new_limits[1])
    ax.set_zlim3d(new_limits[2])

# Luego, después de trazar, llama:
set_axes_equal(ax3d)


print(f"\n🚀 Simulación completada en {time_end - time_start:.2f} segundos")
print(f"📊 Puntos simulados: {steps}")
print(f"⏱️  Tiempo simulado: {time_points[-1]:.1f} años")

# animacion
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
def init():
    ax.clear()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Animación de Órbitas")
    return []

def update(frame):
    ax.clear()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Animación de Órbitas")
    
    for name, traj in trajectories.items():
        if name != "Sol":
            ax.plot(traj[:frame, 0]/1e11, traj[:frame, 1]/1e11, label=name, linewidth=2)
    
    ax.plot(0, 0, 'yo', markersize=10, label='Sol')
    return []

ani = FuncAnimation(fig, update, frames=len(time_points), init_func=init, blit=False, repeat=False, interval=100)
plt.show()


fig3d = plt.figure(figsize=(8, 8))
ax3d = fig3d.add_subplot(111, projection='3d')

ax3d.set_xlim(-3, 3)
ax3d.set_ylim(-3, 3)
ax3d.set_zlim(-0.3, 0.3)
ax3d.set_xlabel("x (UA)")
ax3d.set_ylabel("y (UA)")
ax3d.set_zlabel("z (UA)")
ax3d.set_title("Animación 3D de Órbitas")

# Plot completo y tenue para referencia
for name, traj in trajectories.items():
    if name != "Sol":
        ax3d.plot(traj[:, 0]/1e11, traj[:, 1]/1e11, traj[:, 2]/1e11,
                  color='gray', alpha=0.3, linewidth=1)

# Línea para la animación (una por cuerpo)
lines = {}
for name in trajectories:
    if name != "Sol":
        line, = ax3d.plot([], [], [], label=name, linewidth=2)
        lines[name] = line

sol_point, = ax3d.plot([0], [0], [0], 'yo', markersize=10, label='Sol')
ax3d.legend()
ax3d.grid(True, alpha=0.3)

def update_3d(frame):
    for name, line in lines.items():
        traj = trajectories[name]
        line.set_data(traj[:frame, 0]/1e11, traj[:frame, 1]/1e11)
        line.set_3d_properties(traj[:frame, 2]/1e11)
    return list(lines.values()) + [sol_point]

ani3d = FuncAnimation(fig3d, update_3d, frames=len(time_points), interval=50, blit=False, repeat=False)

plt.show()

plt.close('all')