import numpy as np
import matplotlib.pyplot as plt
from body import Body
from barnes_hut import BarnesHutTree

def main():


    bodies = []

    sol = Body("Sol", 1.989e30, [0, 0, 0], [0, 0, 0], 'yellow')

    
    np.random.seed(42)  # Para reproducibilidad
    asteroid_names = ["Ceres", "Vesta", "Pallas", "Juno", "Astraea", "Hebe", "Iris", "Flora", "Metis", "Eunomia"]
    
    for i, name in enumerate(asteroid_names):
        # Posición aleatoria en el cinturón de asteroides (2.2 - 3.2 UA)
        r = np.random.uniform(2.2e11, 3.2e11)
        theta = np.random.uniform(0, 2*np.pi)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.uniform(-5e10, 5e10)
        
        # Velocidad orbital aproximada
        v_orbit = np.sqrt(6.67430e-11 * sol.mass / r)
        vx = -v_orbit * np.sin(theta)
        vy = v_orbit * np.cos(theta)
        vz = np.random.uniform(-100, 100)  # Pequeña velocidad vertical
        
        mass = np.random.uniform(1e19, 1e21)  # Masa del asteroide
        
        asteroide = Body(name, mass, [x, y, z], [vx, vy, vz], 'gray')
        bodies.append(asteroide)
    
    
    tree = BarnesHutTree(bodies, theta=0.5)
    
    fig = plt.figure(figsize=(18, 12))
    

    ax1 = plt.subplot(2, 3, 1)
    tree.draw_tree_simple(ax1)
    ax1.set_title('Vista Simple')
    
    ax2 = plt.subplot(2, 3, 2)
    tree.draw_tree_simple(ax2, max_depth=3)
    ax2.set_title('Simple (máx. 3 niveles)')
    
    ax3 = plt.subplot(2, 3, 3)
    tree.draw_tree_hierarchical(ax3)
    ax3.set_title('Vista Jerárquica')
    

    ax4 = plt.subplot(2, 3, 4)
    tree.draw_tree_focused(ax4, focus_body_name='Tierra')
    ax4.set_title('Enfoque en Tierra')
    
    ax5 = plt.subplot(2, 3, 5)
    for body in bodies:
        size = 100 if body.name == 'Sol' else (50 if 'planeta' in body.name.lower() else 20)
        ax5.scatter(body.position[0], body.position[1], 
                   color=body.color, s=size, label=body.name)
        ax5.annotate(body.name, (body.position[0], body.position[1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlabel('x (m)')
    ax5.set_ylabel('y (m)')
    ax5.set_title('Solo Cuerpos')

    plt.tight_layout()
    plt.show()

    fig, ax6 = plt.subplots(figsize=(8, 8))
    tree.draw_minimalistic(ax6)
    ax6.set_title('Vista Minimalista')
    ax6.set_xlabel('x (m)')
    ax6.set_ylabel('y (m)')
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper right', fontsize=8)
    plt.show()

    
    

    
    fig3d = plt.figure(figsize=(10, 10))
    ax3d = fig3d.add_subplot(111, projection='3d')
    tree.draw_3D_simple(ax3d)
    ax3d.set_title('Vista 3D del Árbol Barnes-Hut')
    ax3d.set_xlabel('x (m)')
    ax3d.set_ylabel('y (m)')
    ax3d.set_zlabel('z (m)')
    plt.show()


    



if __name__ == "__main__":
    main()