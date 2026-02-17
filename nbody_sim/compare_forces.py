import numpy as np
import matplotlib.pyplot as plt
from barnes_hut import BarnesHutTree
from body import Body

G = 6.67430e-11

def brute_force_force(bodies, target):
    """Calcula la fuerza neta sobre 'target' usando O(N²)."""
    force = np.zeros(3)
    for other in bodies:
        if other is not target:
            r = other.position - target.position
            dist = np.linalg.norm(r)
            if dist > 1e-10:  # Evitar singularidades
                f = G * target.mass * other.mass / dist**2
                force += f * r / dist
    return force

def test(N=50, theta_values=[0.1, 0.3, 0.5, 0.8, 1.0]):
    """Test Barnes-Hut con diferentes parámetros."""
    np.random.seed(42)  # Semilla fija para reproducibilidad
    
    # Generar cuerpos con distribución más realista
    bodies = []
    for i in range(N):
        # Distribución más dispersa
        pos = (np.random.rand(3) - 0.5) * 1000  # -500 a 500
        mass = np.random.lognormal(mean=10, sigma=2)  # Distribución log-normal
        vel = (np.random.rand(3) - 0.5) * 50
        body = Body(f"B{i}", mass, pos, vel, 'blue')
        bodies.append(body)

    print(f"🧪 Testeo con {N} cuerpos...")
    
    # Comparar diferentes valores de theta
    results = {}
    
    for theta in theta_values:
        print(f"\n📐 Theta = {theta}")
        tree = BarnesHutTree(bodies, theta=theta)
        
        errors = []
        times_bh = []
        times_brute = []
        
        for body in bodies:
            # Medir tiempo Barnes-Hut
            import time
            start = time.perf_counter()
            f_bh = tree.compute_force(body)
            times_bh.append(time.perf_counter() - start)
            
            # Medir tiempo fuerza bruta
            start = time.perf_counter()
            f_brute = brute_force_force(bodies, body)
            times_brute.append(time.perf_counter() - start)
            
            # Calcular error
            mag_bh = np.linalg.norm(f_bh)
            mag_brute = np.linalg.norm(f_brute)
            rel_error = np.abs(mag_bh - mag_brute) / (mag_brute + 1e-15)
            errors.append(rel_error)
        
        results[theta] = {
            'errors': errors,
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'time_bh': np.mean(times_bh),
            'time_brute': np.mean(times_brute)
        }
        
        print(f"   Error medio: {results[theta]['mean_error']:.2e}")
        print(f"   Error máximo: {results[theta]['max_error']:.2e}")
        # El speed up es el tiempo que tarda la fuerza bruta dividido por el tiempo que tarda Barnes-Hut. indicando
        # como de rápido es Barnes-Hut comparado con fuerza bruta
        print(f"   Speedup: {results[theta]['time_brute']/results[theta]['time_bh']:.1f}x")
    
    # Visualizaciones mejoradas
    plot_results(results, theta_values, N)
    
    return results

def plot_results(results, theta_values, N):
    """Crear visualizaciones de los resultados."""
    
    # Figura 1: Error vs Theta
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    mean_errors = [results[theta]['mean_error'] for theta in theta_values]
    max_errors = [results[theta]['max_error'] for theta in theta_values]
    
    plt.semilogy(theta_values, mean_errors, 'o-', label='Error medio', color='blue')
    plt.semilogy(theta_values, max_errors, 's-', label='Error máximo', color='red')
    plt.xlabel('Parámetro θ')
    plt.ylabel('Error relativo')
    plt.title('Error vs Parámetro θ')
    plt.legend()
    plt.grid(True)
    
    # Figura 2: Speedup vs Theta
    plt.subplot(2, 2, 2)
    speedups = [results[theta]['time_brute']/results[theta]['time_bh'] for theta in theta_values]
    plt.plot(theta_values, speedups, 'o-', color='green')
    plt.xlabel('Parámetro θ')
    plt.ylabel('Speedup (x veces más rápido)')
    plt.title('Performance vs Parámetro θ')
    plt.grid(True)
    
    # Figura 3: Trade-off Error vs Speedup
    plt.subplot(2, 2, 3)
    plt.scatter(mean_errors, speedups, c=theta_values, cmap='viridis', s=100)
    plt.colorbar(label='θ')
    plt.xlabel('Error medio')
    plt.ylabel('Speedup')
    plt.title('Trade-off: Precisión vs Velocidad')
    plt.xscale('log')
    plt.grid(True)


    # Figura 4: Comparación de tiempos
    plt.subplot(2, 2, 4)
    times_bh = [results[theta]['time_bh'] for theta in theta_values]
    times_brute = [results[theta]['time_brute'] for theta in theta_values]
    bar_width = 0.35
    x = np.arange(len(theta_values))
    plt.bar(x - bar_width/2, times_bh, width=bar_width, label='Barnes-Hut', color='blue')
    plt.bar(x + bar_width/2, times_brute, width=bar_width, label='Fuerza Bruta', color='red')
    plt.xlabel('Parámetro θ')
    plt.ylabel('Tiempo medio (s)')
    plt.title('Comparación de Tiempos')
    plt.xticks(x, theta_values)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'compare_forces_{N}.png', dpi=300)
    # plt.show()
    
   
if __name__ == "__main__":
    N_values = [10, 50, 100, 500, 1000, 2000, 5000, 10000]
    all_results = {}

    for N in N_values:
        print(f"\n{'='*60}\nSimulación con N = {N}\n{'='*60}")
        results = test(N)
        all_results[N] = results

        # Resumen final para cada N
        print("\n" + "="*50)
        print(f"🎯 RESUMEN FINAL para N = {N}:")
        print("="*50)
        
        best_theta = min(results.keys(), key=lambda t: results[t]['mean_error'])
        fastest_theta = max(results.keys(), key=lambda t: results[t]['time_brute']/results[t]['time_bh'])
        
        print(f"Mejor precisión: θ = {best_theta} (error = {results[best_theta]['mean_error']:.2e})")
        print(f"Mejor velocidad: θ = {fastest_theta} (speedup = {results[fastest_theta]['time_brute']/results[fastest_theta]['time_bh']:.1f}x)")
        print(f"Recomendado: θ = 0.3 (buen balance)")

        