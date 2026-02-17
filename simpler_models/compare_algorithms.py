import numpy as np
import matplotlib.pyplot as plt

def calculate_energy(pos, vel, M, G):
    """Calcula energía total (cinética + potencial)"""
    kinetic = 0.5 * np.linalg.norm(vel)**2
    potential = -G * M / np.linalg.norm(pos)
    return kinetic + potential

def load_and_analyze(filename, method_name):
    """Carga trayectoria y calcula conservación de energía"""
    # Cargar posiciones
    positions = np.loadtxt(filename, skiprows=1)
    velocities = np.diff(positions, axis=0) / dt
    
    # Calcular energías
    energies = []
    for i in range(len(velocities)):
        pos = positions[i]
        vel = velocities[i]
        energy = calculate_energy(pos, vel, M, G)
        energies.append(energy)
    
    energies = np.array(energies)
    
    # Estadísticas
    energy_drift = (energies[-1] - energies[0]) / energies[0]
    energy_std = np.std(energies) / np.abs(energies[0])
    
    print(f"{method_name}:")
    print(f"  Deriva energética: {energy_drift:.2e}")
    print(f"  Fluctuación energética: {energy_std:.2e}")
    
    return energies

def analyze_orbit(filename, method_name):
    """Analiza propiedades orbitales"""
    positions = np.loadtxt(filename, skiprows=1)
    
    # Calcular distancias al Sol
    distances = np.linalg.norm(positions, axis=1)
    
    # Estadísticas orbitales
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    min_distance = np.min(distances)
    eccentricity_approx = (max_distance - min_distance) / (max_distance + min_distance)
    
    print(f"{method_name}:")
    print(f"  Distancia promedio: {mean_distance/1.496e11:.3f} UA")
    print(f"  Distancia máxima: {max_distance/1.496e11:.3f} UA")
    print(f"  Distancia mínima: {min_distance/1.496e11:.3f} UA")
    print(f"  Excentricidad aprox: {eccentricity_approx:.6f}")
    print()
    
    return distances

G = 6.67430e-11
M = 1.989e30
dt = 60 * 60 * 24 * 7

# Analizar todos los métodos
energies_euler = load_and_analyze('orbita_euler.txt', 'Euler')
energies_rk4 = load_and_analyze('orbita_rk4.txt', 'RK4')
energies_verlet = load_and_analyze('orbita_verlet.txt', 'Verlet')
energies_leapfrog = load_and_analyze('orbita_leapfrog.txt', 'Leapfrog')

dist_euler = analyze_orbit('orbita_euler.txt', 'Euler')
dist_rk4 = analyze_orbit('orbita_rk4.txt', 'RK4')
dist_verlet = analyze_orbit('orbita_verlet.txt', 'Verlet')
dist_leapfrog = analyze_orbit('orbita_leapfrog.txt', 'Leapfrog')

pos_euler = np.loadtxt('orbita_euler.txt', skiprows=1)
pos_rk4 = np.loadtxt('orbita_rk4.txt', skiprows=1)
pos_verlet = np.loadtxt('orbita_verlet.txt', skiprows=1)
pos_leapfrog = np.loadtxt('orbita_leapfrog.txt', skiprows=1)




# Plot comparativo
# Collage de dos fotos (órbitas completas y zoom final)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Órbitas completas
axs[0].plot(pos_euler[:,0]/1.496e11, pos_euler[:,1]/1.496e11, 'r-', label='Euler', alpha=0.7)
axs[0].plot(pos_rk4[:,0]/1.496e11, pos_rk4[:,1]/1.496e11, 'g-', label='RK4', alpha=0.7)
axs[0].plot(pos_verlet[:,0]/1.496e11, pos_verlet[:,1]/1.496e11, 'b-', label='Verlet', alpha=0.7)
axs[0].plot(pos_leapfrog[:,0]/1.496e11, pos_leapfrog[:,1]/1.496e11, color='orange', label='Leapfrog', alpha=0.7)
axs[0].plot(0, 0, 'yo', markersize=10, label='Sol')
axs[0].set_aspect('equal')
axs[0].set_title('Órbitas completas')
axs[0].set_xlabel('x (UA)')
axs[0].set_ylabel('y (UA)')
axs[0].legend()
axs[0].grid()

# Zoom al final (últimas 100 posiciones)
axs[1].plot(pos_euler[-100:,0]/1.496e11, pos_euler[-100:,1]/1.496e11, 'r-', label='Euler')
axs[1].plot(pos_rk4[-100:,0]/1.496e11, pos_rk4[-100:,1]/1.496e11, 'g-', label='RK4')
axs[1].plot(pos_verlet[-100:,0]/1.496e11, pos_verlet[-100:,1]/1.496e11, 'b-', label='Verlet')
axs[1].plot(pos_leapfrog[-100:,0]/1.496e11, pos_leapfrog[-100:,1]/1.496e11, color='orange', label='Leapfrog')
axs[1].set_title('Zoom: Últimos 100 pasos')
axs[1].set_xlabel('x (UA)')
axs[1].set_ylabel('y (UA)')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()
