import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

G = 6.67430e-11        
M = 1.989e30           
dt = 60 * 60 * 24 * 7  
num_steps = 365 * 200 // 7   
N = 2000         
masa = 5.972e24 * np.ones(N)

angles = np.random.uniform(0, 2*np.pi, N)
radii = np.random.uniform(1.2e11, 2.5e11, N)
z = np.zeros(N)  # Todos en el plano z=0 (disco plano)
# z = np.random.uniform(-0.05e11, 0.05e11, N)  # Pequeña dispersión en z

x = radii * np.cos(angles)
y = radii * np.sin(angles)

pos_init_3d = np.vstack([x, y, z]).T
velocidades_orbitales = np.sqrt(G * M / radii)
vel_init = np.zeros((N, 3))
vel_init[:,0] = -velocidades_orbitales * np.sin(angles)
vel_init[:,1] = velocidades_orbitales * np.cos(angles)
vel_init[:,2] = 0  # Sin velocidad inicial en z

# angles = np.random.uniform(0, 2*np.pi, N)
# radii = np.random.uniform(1e11, 3e11, N)   # más ancho
# pos_init = np.array([radii * np.cos(angles), radii * np.sin(angles)]).T

# velocidades_orbitales = np.sqrt(G * M / radii)
# vel_init = np.array([-velocidades_orbitales * np.sin(angles),
#                      velocidades_orbitales * np.cos(angles)]).T



def acceleration(positions):
    r = -positions
    dist = np.linalg.norm(r, axis=1).reshape(-1,1)
    rhat = r / dist
    return G * M / dist**2 * rhat

def calculate_energy(pos, vel):
    kinetic = 0.5 * np.sum(masa[:,None] * np.sum(vel**2, axis=1).reshape(-1,1))
    potential = -np.sum(G * M * masa / np.linalg.norm(pos, axis=1))
    return kinetic + potential

# --- Simulación Euler ---
pos_euler, vel_euler = pos_init_3d.copy(), vel_init.copy()
positions_euler, energies_euler = [], []
for _ in range(num_steps):
    positions_euler.append(pos_euler.copy())
    energies_euler.append(calculate_energy(pos_euler, vel_euler))
    acc = acceleration(pos_euler)
    pos_euler += vel_euler * dt         # posición primero
    vel_euler += acc * dt              # velocidad después

    #metodo euler-cromer
    # vel_euler += acc * dt
    # pos_euler += vel_euler * dt
positions_euler = np.array(positions_euler)
energies_euler = np.array(energies_euler)

# --- Simulación Leapfrog ---
pos_leap, vel_leap = pos_init_3d.copy(), vel_init.copy()
positions_leap, energies_leap = [], []
acc = acceleration(pos_leap)
for _ in range(num_steps):
    positions_leap.append(pos_leap.copy())
    energies_leap.append(calculate_energy(pos_leap, vel_leap))
    vel_leap += 0.5 * acc * dt
    pos_leap += vel_leap * dt
    acc_new = acceleration(pos_leap)
    vel_leap += 0.5 * acc_new * dt
    acc = acc_new
positions_leap = np.array(positions_leap)
energies_leap = np.array(energies_leap)

# Energía normalizada
energy_diff_euler = (energies_euler - energies_euler[0]) / abs(energies_euler[0])
energy_diff_leap = (energies_leap - energies_leap[0]) / abs(energies_leap[0])
steps = np.arange(num_steps)

# --- Gráficas estáticas estilo paper ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]


### ESTAN PUESTAS LAS POSICIONES INICIALES, SI ESO CAMBIAR A TODAS
# Euler órbitas 
# Euler órbitas en 3D
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(positions_euler[0,:,0], positions_euler[0,:,1], positions_euler[0,:,2], s=1, color="black", alpha=0.7)
ax1.scatter(0, 0, 0, color="yellow", s=100)  # Estrella central
ax1.set_xlim(-2.5e11, 2.5e11)
ax1.set_ylim(-2.5e11, 2.5e11)
ax1.set_zlim(-1e10, 1e10)
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.set_zlabel("z (m)")
ax1.set_title("Euler step (3D)")

# Leapfrog órbitas en 3D
ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(positions_leap[0,:,0], positions_leap[0,:,1], positions_leap[0,:,2], s=1, color="black", alpha=0.7)
ax2.scatter(0, 0, 0, color="yellow", s=100)  # Estrella central
ax2.set_xlim(-2.5e11, 2.5e11)
ax2.set_ylim(-2.5e11, 2.5e11)
ax2.set_zlim(-1e10, 1e10)
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
ax2.set_zlabel("z (m)")
ax2.set_title("Leapfrog step (3D)")

# Energía Euler
ax3.plot(steps, energy_diff_euler, 'k-')
ax3.set_title("Energy (Euler)")
ax3.set_xlabel("Steps")
ax3.set_ylabel("ΔE/E₀")

# Energía Leapfrog
ax4.plot(steps, energy_diff_leap, 'k-')
ax4.set_title("Energy (Leapfrog)")
ax4.set_xlabel("Steps")
ax4.set_ylabel("ΔE/E₀")

# plt.suptitle("Comparison of Euler's and Leapfrog integration energy conserving properties\nfor N bodies orbiting a point source mass. Same time-step used in both simulations.", fontsize=12)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'hspace': 0.18, 'wspace': 0.18})
ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

# Minimalist colors and styles
color_euler = "#242323"
color_leap = "#242323"
star_color = "#000000"
bg_color = "#FFFFFF"
trail_alpha = 0.0  # No trails for minimalism

for ax, metodo, color in zip([ax1, ax2], ["Euler", "Leapfrog"], [color_euler, color_leap]):
    ax.set_facecolor(bg_color)
    ax.set_xlim(-5e11, 5e11)
    ax.set_ylim(-5e11, 5e11)
    ax.set_aspect("equal")
    ax.plot(0, 0, 'o', color=star_color, markersize=10, markeredgewidth=0, zorder=10)
    ax.set_title(metodo, fontsize=15, fontweight='bold', color=color)
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.tick_params(colors='#888888', labelsize=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

points_euler = ax1.plot([], [], 'o', color=color_euler, markersize=3, alpha=0.8)[0]
trail_euler, = ax1.plot([], [], '-', color=color_euler, alpha=trail_alpha, lw=1)
points_leap = ax2.plot([], [], 'o', color=color_leap, markersize=3, alpha=0.8)[0]
trail_leap, = ax2.plot([], [], '-', color=color_leap, alpha=trail_alpha, lw=1)

for ax, color, energy_diff, title in zip([ax3, ax4], [color_euler, color_leap], [energy_diff_euler, energy_diff_leap], ["Energy (Euler)", "Energy (Leapfrog)"]):
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, num_steps)
    ax.set_ylim(np.min(energy_diff)*1.2, np.max(energy_diff)*1.2)
    ax.set_title(title, fontsize=14, fontweight='bold', color=color)
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("ΔE/E₀", fontsize=12)
    ax.tick_params(colors='#888888', labelsize=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

energy_line_euler, = ax3.plot([], [], '-', color=color_euler, lw=1.5)
energy_line_leap, = ax4.plot([], [], '-', color=color_leap, lw=1.5)

trail_length = 0  # No trail for minimalism

def init():
    points_euler.set_data([], [])
    trail_euler.set_data([], [])
    points_leap.set_data([], [])
    trail_leap.set_data([], [])
    energy_line_euler.set_data([], [])
    energy_line_leap.set_data([], [])
    return points_euler, trail_euler, points_leap, trail_leap, energy_line_euler, energy_line_leap

def update(frame):
    x_euler = positions_euler[frame][:,0]
    y_euler = positions_euler[frame][:,1]
    points_euler.set_data(x_euler, y_euler)
    start = max(0, frame-trail_length)
    trail_euler.set_data([], [])

    x_leap = positions_leap[frame][:,0]
    y_leap = positions_leap[frame][:,1]
    points_leap.set_data(x_leap, y_leap)
    trail_leap.set_data([], [])

    energy_line_euler.set_data(steps[:frame+1], energy_diff_euler[:frame+1])
    energy_line_leap.set_data(steps[:frame+1], energy_diff_leap[:frame+1])

    return points_euler, trail_euler, points_leap, trail_leap, energy_line_euler, energy_line_leap

fig.patch.set_facecolor(bg_color)
frame_skip = 2
ani = FuncAnimation(fig, update, frames=range(0, num_steps, frame_skip),
                    init_func=init, blit=True, interval=8)

plt.tight_layout(rect=[0,0,1,0.96])
plt.show()


# Ahora lo mismo pero en 2D (el plot inicial)
# Collage de 2 fotos: órbitas Euler y Leapfrog en 2D
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# ax1, ax2 = axes

# # Euler órbitas en 2D
# ax1.scatter(positions_euler[:,:,0], positions_euler[:,:,1], s=1, color="black", alpha=0.7)
# ax1.scatter(0, 0, color="yellow", s=100)  # Estrella central
# ax1.set_xlim(-2.5e11, 2.5e11)
# ax1.set_ylim(-2.5e11, 2.5e11)
# ax1.set_xlabel("x (m)")
# ax1.set_ylabel("y (m)")
# ax1.set_title("Euler step (2D)")

# # Leapfrog órbitas en 2D
# ax2.scatter(positions_leap[:,:,0], positions_leap[:,:,1], s=1, color="black", alpha=0.7)
# ax2.scatter(0, 0, color="yellow", s=100)  # Estrella central
# ax2.set_xlim(-2.5e11, 2.5e11)
# ax2.set_ylim(-2.5e11, 2.5e11)
# ax2.set_xlabel("x (m)")
# ax2.set_ylabel("y (m)")
# ax2.set_title("Leapfrog step (2D)")

# plt.tight_layout()
# plt.show()


# Animación 3D comparando órbitas Euler y Leapfrog

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

euler_scatter = ax.scatter([], [], [], s=1, color="blue", alpha=0.6, label="Euler")
leap_scatter = ax.scatter([], [], [], s=1, color="red", alpha=0.6, label="Leapfrog")
star_scatter = ax.scatter(0, 0, 0, color="yellow", s=100, label="Star")

# Trails para todas las partículas
euler_trails = [ax.plot([], [], [], color="blue", alpha=0.3, lw=1)[0] for _ in range(N)]
leap_trails = [ax.plot([], [], [], color="red", alpha=0.3, lw=1)[0] for _ in range(N)]

ax.set_xlim(-6e11, 6e11)
ax.set_ylim(-6e11, 6e11)
ax.set_zlim(-1e10, 1e10)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.set_title("Euler vs Leapfrog Orbits (3D)")
ax.legend(loc="upper right")

trail_length = 5  # número de frames para el trail

ax.view_init(elev=30, azim=45)
ax.grid(True, alpha=0.2)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

def init_anim():
    euler_scatter._offsets3d = ([], [], [])
    leap_scatter._offsets3d = ([], [], [])
    for trail in euler_trails:
        trail.set_data([], [])
        trail.set_3d_properties([])
    for trail in leap_trails:
        trail.set_data([], [])
        trail.set_3d_properties([])
    return [euler_scatter, leap_scatter, star_scatter] + euler_trails + leap_trails

def update_anim(frame):
    euler_x = positions_euler[frame,:,0]
    euler_y = positions_euler[frame,:,1]
    euler_z = positions_euler[frame,:,2]
    leap_x = positions_leap[frame,:,0]
    leap_y = positions_leap[frame,:,1]
    leap_z = positions_leap[frame,:,2]
    euler_scatter._offsets3d = (euler_x, euler_y, euler_z)
    leap_scatter._offsets3d = (leap_x, leap_y, leap_z)

    start = max(0, frame-trail_length)
    for i in range(N):
        euler_trails[i].set_data(positions_euler[start:frame+1,i,0], positions_euler[start:frame+1,i,1])
        euler_trails[i].set_3d_properties(positions_euler[start:frame+1,i,2])
        leap_trails[i].set_data(positions_leap[start:frame+1,i,0], positions_leap[start:frame+1,i,1])
        leap_trails[i].set_3d_properties(positions_leap[start:frame+1,i,2])

    return [euler_scatter, leap_scatter, star_scatter] + euler_trails + leap_trails

ani = FuncAnimation(fig, update_anim, frames=range(0, num_steps, 2),
                    init_func=init_anim, blit=False, interval=10)

plt.tight_layout()

# DESCOMENTAR SI SE EJECUTA DESDE WSL ( y si se quiere guardar, claro )!!
# ani.save("orbits_animation.mp4", writer='ffmpeg', fps=30)
plt.show()
