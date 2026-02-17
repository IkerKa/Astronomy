### PENDIENTE

import time
from body import Body
from simulator import Simulator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

G = 6.67430e-11  # Constante gravitacional
c = 3e8

## Para que dos galaxias choquen debe haber disipación.

# Crear simulador
# sim = Simulator(use_barnes_hut=True, theta=0.5)  
# bodies = []

# ## Galaxy
# m_bh = 1e30
# d = 3e12  # separación entre agujeros negros (~20 UA)
# Mtot = 2 * m_bh # Masa total
# omega = np.sqrt(G * Mtot / d**3)
# v_bh = omega * (d / 2)  # velocidad de cada BH alrededor del baricentro
# v_scale = 0.8  # para que se acerquen
# v_bh *= v_scale

# # Posiciones/velocidades iniciales de los BH (órbita circular)
# bh1_pos = np.array([-d/2, 0.0, 0.0])
# bh2_pos = np.array([ d/2, 0.0, 0.0])
# bh1_vel = np.array([0.0,  v_bh, 0.0])
# bh2_vel = np.array([0.0, -v_bh, 0.0])

# # Massive black hole A
# black_hole = Body("Black Hole", m_bh, bh1_pos.tolist(), bh1_vel.tolist(), color='black')
# bodies.append(black_hole)
# sim.add_body(black_hole)

# # N planetas orbitando el BH1 (velocidad local + velocidad del BH)
# N = 15
# for i in range(N):
#     r = np.random.uniform(5e10, 2e11) 
#     v = np.sqrt(G * black_hole.mass / r)
#     angle = np.random.uniform(0, 2*np.pi)
#     x_local = r * np.cos(angle)
#     y_local = r * np.sin(angle)
#     vx_local = -v * np.sin(angle)
#     vy_local =  v * np.cos(angle)
#     pos = bh1_pos + np.array([x_local, y_local, 0.0])
#     vel = bh1_vel + np.array([vx_local, vy_local, 0.0])
#     planet = Body(f"Planet {i+1}", 1e24, pos.tolist(), vel.tolist(), color='blue')
#     bodies.append(planet)
#     sim.add_body(planet)

# #Another galaxy next to this one
# black_hole_2 = Body("Black Hole 2", m_bh, bh2_pos.tolist(), bh2_vel.tolist(), color='black')
# bodies.append(black_hole_2)
# sim.add_body(black_hole_2)

# for i in range(N):
#     r = np.random.uniform(5e10, 2e11) 
#     v = np.sqrt(G * black_hole_2.mass / r)
#     angle = np.random.uniform(0, 2*np.pi)
#     x_local = r * np.cos(angle)
#     y_local = r * np.sin(angle)
#     vx_local = -v * np.sin(angle)
#     vy_local =  v * np.cos(angle)
#     pos = bh2_pos + np.array([x_local, y_local, 0.0])
#     vel = bh2_vel + np.array([vx_local, vy_local, 0.0])
#     planet = Body(f"Planet 2.{i+1}", 1e24, pos.tolist(), vel.tolist(), color='red')
#     bodies.append(planet)
#     sim.add_body(planet)


# # Posición aleatoria para el tercer agujero negro entre los otros dos
# # Tercer agujero negro: órbita excéntrica y desplazada, con planetas en disco inclinado
# # Posición inicial: más lejos y fuera del plano principal
# bh3_pos = np.array([0.0, 1.2*d, 0.8*d])
# # Velocidad inicial: movimiento hacia el centro, con componente vertical
# bh3_vel = np.array([0.7*v_bh, -0.4*v_bh, 0.5*v_bh])
# black_hole_3 = Body("Black Hole 3", m_bh, bh3_pos.tolist(), bh3_vel.tolist(), color='purple')
# bodies.append(black_hole_3)
# sim.add_body(black_hole_3)

# # Planetas en disco inclinado (30 grados respecto al plano xy)
# N3 = N
# inclination = np.deg2rad(30)
# for i in range(N3):
#     r = np.random.uniform(5e10, 2e11)
#     v = np.sqrt(G * black_hole_3.mass / r)
#     angle = np.random.uniform(0, 2*np.pi)
#     # Coordenadas en disco inclinado
#     x_local = r * np.cos(angle)
#     y_local = r * np.sin(angle) * np.cos(inclination)
#     z_local = r * np.sin(angle) * np.sin(inclination)
#     # Velocidad tangencial en disco inclinado
#     vx_local = -v * np.sin(angle)
#     vy_local = v * np.cos(angle) * np.cos(inclination)
#     vz_local = v * np.cos(angle) * np.sin(inclination)
#     pos = bh3_pos + np.array([x_local, y_local, z_local])
#     vel = bh3_vel + np.array([vx_local, vy_local, vz_local])
#     planet = Body(f"Planet 3.{i+1}", 1e24, pos.tolist(), vel.tolist(), color='purple')
#     bodies.append(planet)
#     sim.add_body(planet)


sim = Simulator(use_barnes_hut=True, theta=0.5)
bodies = []

# Constantes
m_bh1 = 5e30   # BH central más masivo
m_bh2 = 1e30   # BH secundario
m_bh3 = 1e30   # BH tercero inclinado
d = 1e12       # distancia inicial entre BH1 y BH2 (~7 UA)

# Velocidad orbital reducida para colisión lenta
Mtot = m_bh1 + m_bh2
omega = np.sqrt(G * Mtot / d**3)
v_orb = omega * (d / 2)
v_scale = 0.65  # < 0.8 para hacer órbitas excéntricas
v_orb *= v_scale

# BH1 (masivo)
bh1_pos = np.array([-d/2, 0.0, 0.0])
bh1_vel = np.array([0.0,  v_orb, 0.0])
bh1 = Body("BH1", m_bh1, bh1_pos.tolist(), bh1_vel.tolist(), color='black')
bodies.append(bh1)
sim.add_body(bh1)

# BH2
bh2_pos = np.array([ d/2, 0.0, 0.0])
bh2_vel = np.array([0.0, -v_orb, 0.0])
bh2 = Body("BH2", m_bh2, bh2_pos.tolist(), bh2_vel.tolist(), color='gray')
bodies.append(bh2)
sim.add_body(bh2)

# BH3 (inclinación 45°, más alejado y cayendo hacia el centro)
bh3_pos = np.array([0.0, 2*d, d])
bh3_vel = np.array([0.5*v_orb, -0.3*v_orb, -0.4*v_orb])
bh3 = Body("BH3", m_bh3, bh3_pos.tolist(), bh3_vel.tolist(), color='purple')
bodies.append(bh3)
sim.add_body(bh3)

# Función para generar discos planetarios
def add_disk(central_body, N, mass, rmin, rmax, inclination, color):
    for i in range(N):
        r = np.random.uniform(rmin, rmax)
        v = np.sqrt(G * central_body.mass / r)
        angle = np.random.uniform(0, 2*np.pi)

        # Coordenadas en disco inclinado
        x_local = r * np.cos(angle)
        y_local = r * np.sin(angle) * np.cos(inclination)
        z_local = r * np.sin(angle) * np.sin(inclination)

        # Velocidad tangencial
        vx_local = -v * np.sin(angle)
        vy_local =  v * np.cos(angle) * np.cos(inclination)
        vz_local =  v * np.cos(angle) * np.sin(inclination)

        pos = np.array(central_body.position) + np.array([x_local, y_local, z_local])
        vel = np.array(central_body.velocity) + np.array([vx_local, vy_local, vz_local])

        planet = Body(f"{central_body.name}-P{i+1}", mass, pos.tolist(), vel.tolist(), color=color)
        bodies.append(planet)
        sim.add_body(planet)

# Discos
add_disk(bh1, 5, 1e22, 5e10, 2e11, np.deg2rad(0), "blue")     # fino, plano
add_disk(bh2, 5, 1e22, 5e10, 2e11, np.deg2rad(30), "red")     # inclinado
add_disk(bh3, 30, 1e22, 5e10, 3e11, np.deg2rad(45), "green")   # disperso e inclinado




print(f"🌌 Simulando {len(bodies)} cuerpos celestes...")


#dt de un dia
dt = 60 * 60 * 24  # Un día
steps = 365 * 5

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

AU = 1.496e11
traj_AU = {name: traj / AU for name, traj in trajectories.items()}

x_min = min(tr[:, 0].min() for tr in traj_AU.values())
x_max = max(tr[:, 0].max() for tr in traj_AU.values())
y_min = min(tr[:, 1].min() for tr in traj_AU.values())
y_max = max(tr[:, 1].max() for tr in traj_AU.values())
padx = 0.05 * max(1e-6, x_max - x_min)
pady = 0.05 * max(1e-6, y_max - y_min)

# Gráfico 2D mejorado: trayectorias con líneas, puntos finales destacados y etiquetas
fig, ax = plt.subplots(figsize=(8, 8))
for body in bodies:
    tr = traj_AU[body.name]
    # Dibuja la trayectoria con una línea tenue
    ax.plot(
        tr[:, 0], tr[:, 1],
        linestyle='-', linewidth=1,
        color=body.color if hasattr(body, "color") else ("k" if "Black Hole" in body.name else "gray"),
        alpha=0.5
    )
    # Dibuja el punto final más grande y visible
    ax.plot(
        tr[-1, 0], tr[-1, 1],
        marker='o',
        markersize=8 if "Black Hole" in body.name else 4,
        color=body.color if hasattr(body, "color") else ("k" if "Black Hole" in body.name else "gray"),
        label=body.name
    )
ax.set_xlabel("x (UA)")
ax.set_ylabel("y (UA)")
ax.set_title("Órbitas 2D Mejoradas")
ax.set_aspect('equal')
ax.set_xlim(x_min - padx, x_max + padx)
ax.set_ylim(y_min - pady, y_max + pady)
ax.grid(False)
plt.tight_layout()
plt.show()

# 2D animation con trayectorias cortas (cola con alpha variable)
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_xlabel("x (UA)")
ax.set_ylabel("y (UA)")
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.plot(0, 0, marker='+', color='k', markersize=10, linestyle='None')

points = []
tails = []
tail_length = 200  # frames de cola

# Cola especial para los black holes (más larga)
bh_tail_length = 300

for body in bodies:
    ms = 6 if "Black Hole" in body.name else 2
    col = body.color if hasattr(body, "color") else ("k" if "Black Hole" in body.name else "gray")
    p, = ax.plot([], [], marker='o', linestyle='None', markersize=ms, color=col)
    points.append(p)
    # Usamos LineCollection para alpha variable
    lc = LineCollection([], linewidths=2 if "Black Hole" in body.name else 1, colors=[col])
    tails.append(lc)
    ax.add_collection(lc)

n_frames = min(traj.shape[0] for traj in traj_AU.values())

def update(frame):
    for i, body in enumerate(bodies):
        tr = traj_AU[body.name]
        points[i].set_data([tr[frame, 0]], [tr[frame, 1]])
        if "Black Hole" in body.name:
            tail_len = bh_tail_length
        else:
            tail_len = tail_length
        start = max(0, frame - tail_len)
        tail_traj = tr[start:frame+1]
        if len(tail_traj) > 1:
            # Use only x and y coordinates for segments
            tail_traj_2d = tail_traj[:, :2]
            segments = np.array([tail_traj_2d[j:j+2] for j in range(len(tail_traj_2d)-1)])
            # Alpha decae exponencialmente desde 1 a 0.05
            alphas = np.linspace(1.0, 0.05, len(segments))
            # Usamos el color del body y cambiamos alpha
            base_color = mcolors.to_rgba(body.color if hasattr(body, "color") else ("k" if "Black Hole" in body.name else "gray"))
            colors = np.zeros((len(segments), 4))
            colors[:, :3] = base_color[:3]
            colors[:, 3] = alphas
            tails[i].set_segments(segments)
            tails[i].set_color(colors)
        else:
            tails[i].set_segments([])
    return points + tails

skip = 5
ani = FuncAnimation(fig, update, frames=range(0, n_frames, skip), interval=10)
# ani.save("3galaxies.mp4", writer='ffmpeg', fps=30)

plt.tight_layout()
plt.show()
plt.close()

# 3D plot de trayectorias y posiciones finales
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Limites automáticos en AU
x_min3d = min(tr[:, 0].min() for tr in traj_AU.values())
x_max3d = max(tr[:, 0].max() for tr in traj_AU.values())
y_min3d = min(tr[:, 1].min() for tr in traj_AU.values())
y_max3d = max(tr[:, 1].max() for tr in traj_AU.values())
z_min3d = min(tr[:, 2].min() for tr in traj_AU.values())
z_max3d = max(tr[:, 2].max() for tr in traj_AU.values())
padx3d = 0.05 * max(1e-6, x_max3d - x_min3d)
pady3d = 0.05 * max(1e-6, y_max3d - y_min3d)
padz3d = 0.05 * max(1e-6, z_max3d - z_min3d)

ax.set_xlim(x_min3d - padx3d, x_max3d + padx3d)
ax.set_ylim(y_min3d - pady3d, y_max3d + pady3d)
ax.set_zlim(z_min3d - padz3d, z_max3d + padz3d)
ax.set_xlabel("x (UA)")
ax.set_ylabel("y (UA)")
ax.set_zlabel("z (UA)")
ax.set_title("Órbitas 3D Mejoradas")

# Trayectorias y puntos finales
for body in bodies:
    tr = traj_AU[body.name]
    ax.plot(
        tr[:, 0], tr[:, 1], tr[:, 2],
        linestyle='-', linewidth=1,
        color=body.color if hasattr(body, "color") else ("k" if "Black Hole" in body.name else "gray"),
        alpha=0.5
    )
    ax.scatter(
        tr[-1, 0], tr[-1, 1], tr[-1, 2],
        marker='o',
        s=80 if "Black Hole" in body.name else 30,
        color=body.color if hasattr(body, "color") else ("k" if "Black Hole" in body.name else "gray"),
        label=body.name
    )

ax.legend(loc="upper right")
ax.view_init(elev=30, azim=45)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# Animación 3D de trayectorias con cola
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("x (UA)")
ax.set_ylabel("y (UA)")
ax.set_zlabel("z (UA)")
ax.set_title("Animación 3D de Órbitas")

ax.set_xlim(x_min3d - padx3d, x_max3d + padx3d)
ax.set_ylim(y_min3d - pady3d, y_max3d + pady3d)
ax.set_zlim(z_min3d - padz3d, z_max3d + padz3d)
ax.view_init(elev=30, azim=45)
ax.grid(True, alpha=0.2)

points3d = []
tails3d = []
tail_length3d = 200
bh_tail_length3d = 300

for body in bodies:
    ms = 60 if "Black Hole" in body.name else 20
    col = body.color if hasattr(body, "color") else ("k" if "Black Hole" in body.name else "gray")
    p = ax.scatter([], [], [], s=ms, color=col)
    points3d.append(p)
    # Cola como línea 3D
    t, = ax.plot([], [], [], color=col, alpha=0.5, lw=2 if "Black Hole" in body.name else 1)
    tails3d.append(t)

n_frames3d = min(traj.shape[0] for traj in traj_AU.values())

def update3d(frame):
    for i, body in enumerate(bodies):
        tr = traj_AU[body.name]
        x, y, z = tr[frame, 0], tr[frame, 1], tr[frame, 2]
        points3d[i]._offsets3d = ([x], [y], [z])
        tail_len = bh_tail_length3d if "Black Hole" in body.name else tail_length3d
        start = max(0, frame - tail_len)
        tail_traj = tr[start:frame+1]
        if len(tail_traj) > 1:
            tails3d[i].set_data(tail_traj[:, 0], tail_traj[:, 1])
            tails3d[i].set_3d_properties(tail_traj[:, 2])
        else:
            tails3d[i].set_data([], [])
            tails3d[i].set_3d_properties([])
    return points3d + tails3d

skip3d = 5
ani3d = FuncAnimation(fig, update3d, frames=range(0, n_frames3d, skip3d), interval=10)
# ani3d.save("3galaxies_3d.mp4", writer='ffmpeg', fps=30)

plt.tight_layout()
plt.show()
plt.close()