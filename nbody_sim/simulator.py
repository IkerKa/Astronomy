import numpy as np
from body import Body
from barnes_hut import BarnesHutTree
from matplotlib import pyplot as plt

### TODO: IMPLEMENTAR BARNES-HUT (primero estudiarlo)
### TODO: IMPLEMENTAR el step de RK4
### TODO: IMPLEMENTAR el step de Verlet (estudiarlo) y compararlos
### TODO: IMPLEMENTAR el step de Leapfrog (estudiarlo) y compararlos
### TODO: Comparar todos los métodos de integración (Euler, RK4, Verlet, Leapfrog) y temas de perdida de energía y eso

G = 6.67430e-11  # Constante gravitacional

class Simulator:

    def __init__(self, use_barnes_hut=False, theta=0.5):
        self.use_barnes_hut = use_barnes_hut
        self.theta = theta
        self.bodies = []
        self.tree = None


    def add_body(self, body):
        self.bodies.append(body)

    def build_tree_if_needed(self):
        if self.use_barnes_hut:
            self.tree = BarnesHutTree(self.bodies)
  


    def compute_acceleration(self, target_body):
        if self.use_barnes_hut:
            if self.tree is None:
                raise ValueError("El árbol Barnes-Hut no ha sido inicializado.")
            
            force = self.tree.compute_force(target_body)
            acceleration = force / target_body.mass
            
            
            return acceleration
        
        total_acc = np.zeros(3)  # aceleracion 3D ax,ay,az

        for other in self.bodies:
            if other is target_body:
                continue # no se afecta a sí mismo


            r_vec = other.position - target_body.position
            distance = np.linalg.norm(r_vec)

            if distance == 0:
                continue 


            # recordamos, a = F / m, donde m se cancela con la ley de gravitacion universal
            acc = G * other.mass * r_vec / distance**3        # explicado en el cuaderno

            total_acc += acc

        return total_acc
    
    def euler_step(self, dt):
        """
        Avanzamos la simulación un paso de tiempo empleando el método de Euler
        """
        self.build_tree_if_needed()  # Aseguramos que el árbol está construido si se usa Barnes-Hut

        accelerations = []

        # 1. como en el ejemplo simple calculamos las aceleraciones, pero de todos en este caso
        for bd in self.bodies:
            acc = self.compute_acceleration(bd)
            accelerations.append(acc)

        # 2. ahora que tenemos las aceleraciones, actualizamos velocidad y posicion con el método de euler
        for i, body in enumerate(self.bodies):
            curr_acc = accelerations[i]
            body.velocity += curr_acc * dt
            body.position += body.velocity *dt


    def rk4_step(self, dt):
        """
        Avanzamos la simulación un paso de tiempo empleando el método de RK4
        """

        self.build_tree_if_needed()  # Aseguramos que el árbol está construido si se usa Barnes-Hut
        
        # Guardar estado inicial
        initial_positions = [body.position.copy() for body in self.bodies]
        initial_velocities = [body.velocity.copy() for body in self.bodies]
        
        # k1
        k1_v = [self.compute_acceleration(body) for body in self.bodies]
        k1_r = [body.velocity.copy() for body in self.bodies]
        
        # k2
        for i, body in enumerate(self.bodies):
            body.position = initial_positions[i] + dt/2 * k1_r[i]
            body.velocity = initial_velocities[i] + dt/2 * k1_v[i]
        
        k2_v = [self.compute_acceleration(body) for body in self.bodies]
        k2_r = [body.velocity.copy() for body in self.bodies]

        # k3
        for i, body in enumerate(self.bodies):
            body.position = initial_positions[i] + dt/2 * k2_r[i]
            body.velocity = initial_velocities[i] + dt/2 * k2_v[i]
        
        k3_v = [self.compute_acceleration(body) for body in self.bodies]
        k3_r = [body.velocity.copy() for body in self.bodies]

        # k4
        for i, body in enumerate(self.bodies):
            body.position = initial_positions[i] + dt * k3_r[i]
            body.velocity = initial_velocities[i] + dt * k3_v[i]
        
        k4_v = [self.compute_acceleration(body) for body in self.bodies]
        k4_r = [body.velocity.copy() for body in self.bodies]
        
        for i, body in enumerate(self.bodies):
            body.position = initial_positions[i] + dt/6 * (k1_r[i] + 2*k2_r[i] + 2*k3_r[i] + k4_r[i])
            body.velocity = initial_velocities[i] + dt/6 * (k1_v[i] + 2*k2_v[i] + 2*k3_v[i] + k4_v[i]) 


    def verlet_step(self, dt):
        """
        Avanzamos la simulación un paso de tiempo empleando el método de Verlet
        """

        self.build_tree_if_needed()  # Aseguramos que el árbol está construido si se usa Barnes-Hut

        initial_accelerations = []

        # Calculamos todas las aceleraciones iniciales
        for bd in self.bodies:
            acc = self.compute_acceleration(bd)
            initial_accelerations.append(acc)


        #cogemos las posiciones iniciales
        initial_positions = [body.position.copy() for body in self.bodies]
        initial_velocities = [body.velocity.copy() for body in self.bodies]

        # Antes de computar la nueva acceleracion, actualizamos las posiciones
        for i, body in enumerate(self.bodies):
            curr_acc = initial_accelerations[i]
            body.position = initial_positions[i] + body.velocity * dt + (curr_acc * dt**2) / 2


        # ahora que tenemos todos los cuerpos en sus nuevas posiciones, calculamos las nuevas aceleraciones
        new_accelerations = []
        for body in self.bodies:
            new_acc = self.compute_acceleration(body)
            new_accelerations.append(new_acc)

        # con las nuevas aceleraciones, calculamos las nuevas velocidades
        for i, body in enumerate(self.bodies):
            body.velocity = initial_velocities[i] + 0.5 * (initial_accelerations[i] + new_accelerations[i]) * dt

    def leapfrog_step(self, dt):
        """
        Avanzamos la simulación un paso de tiempo empleando el método de Leapfrog
        """

        self.build_tree_if_needed()  # Aseguramos que el árbol está construido si se usa Barnes-Hut

        
        initial_accelerations = []

        # Calculamos todas las aceleraciones iniciales
        for bd in self.bodies:
            acc = self.compute_acceleration(bd)
            initial_accelerations.append(acc)


        #cogemos las posiciones iniciales
        initial_positions = [body.position.copy() for body in self.bodies]
        initial_velocities = [body.velocity.copy() for body in self.bodies]


        # Actualizamos las velocidades al medio paso
        for i, body in enumerate(self.bodies):
            curr_acc = initial_accelerations[i]
            body.velocity = initial_velocities[i] + 0.5 * curr_acc * dt

        # Ahora que tenemos la primera velocidad a medio paso, actualizamos las posiciones
        for i, body in enumerate(self.bodies):
            body.position = initial_positions[i] + body.velocity * dt

        # Calculamos las nuevas aceleraciones
        new_accelerations = []
        for body in self.bodies:
            new_acc = self.compute_acceleration(body)
            new_accelerations.append(new_acc)

        # Actualizamos las velocidades al final del paso
        for i, body in enumerate(self.bodies):
            body.velocity += 0.5 * new_accelerations[i] * dt
