from turtle import speed
from body import Body
from simulator import Simulator
from visualizer import Visualizer
from vpython import rate, color, text, vector, scene, box, sphere, arrow, cylinder, keysdown, canvas
import numpy as np

class AdvancedSimulation:
    def __init__(self):
        self.sim = Simulator()
        self.setup_solar_system()
        # self.setup_asteroids()
        self.viz = Visualizer(self.sim.bodies)
        self.time_scale = 1.0
        self.total_time = 0.0
        self.paused = False  # Mover aquí para mejor organización
        self.speed_factor = 1.0
        self.current_focus = None

    def setup_asteroids(self):
        G = 6.67430e-11
        sun_mass = 1.989e30

        sun = Body("Sol ☀️", sun_mass, [0, 0, 0], [0, 0, 0], 'yellow', radius=6.96e8)
        
        # Distancias y posiciones en 3D
        positions = [
            [2e11, 0, 1e10],
            [-2e11, 0, -1.5e10],
            [0, 2e11, 0.5e10]
        ]
        
        asteroides = []
        for i, pos in enumerate(positions):
            r = np.linalg.norm(pos)  # distancia al sol
            v_mag = np.sqrt(G * sun_mass / r)  # velocidad orbital circular
            
            # Vector de velocidad perpendicular a la posición
            # Usamos una simple rotación 90° en el plano XY y componente Z proporcional
            vel = np.cross(pos, [0,0,1])
            vel = vel / np.linalg.norm(vel) * v_mag  # normalizar y escalar
            vel[2] = v_mag * 0.6 * ((-1)**i)  # componente Z pequeña para órbita 3D

            rand_color = np.random.choice(['red', 'green', 'blue', 'yellow', 'orange'])
            asteroides.append(Body(f"Asteroide {i+1}", 1e12, pos, vel, rand_color, radius=1e6))

        self.sim.add_body(sun)
        for asteroide in asteroides:
            self.sim.add_body(asteroide)


        self.current_method = 'verlet'  # Método por defecto

    def setup_solar_system(self):
        """Configurar sistema solar con datos más precisos"""
        # Sol
        sol = Body("Sol ☀️", 1.989e30, [0, 0, 0], [0, 0, 0], 'yellow', radius=6.96e8)

        # Planetas en orden desde el Sol
        mercurio = Body("Mercurio ☿️", 3.301e23,
                       [5.79e10, 0, 0],
                       [0, 47.87e3, 0],
                       'white',
                       radius=2.44e6)

        venus = Body("Venus ♀️", 4.867e24,
                    [1.082e11, 0, 0],
                    [0, 35.02e3, 0],
                    'orange',
                    radius=6.0518e6)

        tierra = Body("Tierra 🌍", 5.972e24, 
                     [1.496e11, 0, 0],
                     [0, 29.78e3, 0],
                     'blue',
                     radius=6.4e6)
        
        marte = Body("Marte 🔴", 6.417e23, 
                    [2.279e11, 0, 0], 
                    [0, 24.077e3, 0],
                    'red',
                    radius=3.4e6)

        jupiter = Body("Júpiter 🪐", 1.898e27,
                      [7.785e11, 0, 0],
                      [0, 13.07e3, 0],
                      'brown',
                      radius=6.9911e7)

        # Añadir en orden
        for body in [sol, mercurio, venus, tierra, marte, jupiter]:
            self.sim.add_body(body)

        self.current_method = 'verlet'

    def focus_on_body(self, name):
        """Enfocar la cámara en un cuerpo por su nombre"""
        for body in self.sim.bodies:
            if body.name.startswith(name): 
                self.current_focus = body  # Guardar el cuerpo enfocado
                scene.center = vector(*body.position)
                print(f"🔍 Enfocando en {body.name}")
                return
        print(f"❌ Cuerpo '{name}' no encontrado")


    def handle_input(self):
        """Manejar entrada del usuario"""
        if len(keysdown()) > 0:
            key = keysdown()[0]

            if key == ' ':
                self.paused = not self.paused
                print("⏸️ Pausado" if self.paused else "▶️ Reanudado")
                while ' ' in keysdown():
                    rate(30)

            elif key == '+' or key == '=':
                self.speed_factor *= 1.2
                print(f"🚀 Velocidad: {self.speed_factor:.1f}x")
                
            elif key == '-':
                self.speed_factor /= 1.2
                print(f"🐢 Velocidad: {self.speed_factor:.1f}x")

            # elif key == 'r':
            #     print("🔄 Reiniciando simulación...")
            #     self.total_time = 0.0
            #     self.current_focus = None  # Reset del enfoque
            #     self.setup_solar_system()
            #     self.viz = Visualizer(self.sim.bodies)

            elif key == '0':  # Enfocar en el Sol
                self.focus_on_body("Sol")
            elif key == '1':
                self.focus_on_body("Mercurio")
            elif key == '2':
                self.focus_on_body("Venus")
            elif key == '3':
                self.focus_on_body("Tierra")
            elif key == '4':
                self.focus_on_body("Marte")
            elif key == '5':
                self.focus_on_body("Júpiter")




    def step(self, dt):
        """Realizar un paso de simulación según el método actual"""
        if self.current_method == 'euler':
            self.sim.euler_step(dt * self.time_scale)
        elif self.current_method == 'rk4':
            self.sim.rk4_step(dt * self.time_scale)
        elif self.current_method == 'verlet':
            self.sim.verlet_step(dt * self.time_scale)
        elif self.current_method == 'leapfrog':
            self.sim.leapfrog_step(dt * self.time_scale)
        else:
            raise ValueError(f"Método desconocido: {self.current_method}")

    def print_status(self):
        """Mostrar información del estado actual"""
        years = self.total_time / (365.25 * 24 * 3600)
        if int(years) % 5 == 0 and years > 0:  # Cada 5 años
            print(f"📅 Tiempo transcurrido: {years:.1f} años")

    def run(self):
        """Ejecutar simulación con controles"""
        dt = 60 * 60 * 24 * 7  # 1 semana
        steps = 365 * 100  # 100 años
        
        print("🚀 Simulación del Sistema Solar")
        print("Controles:")
        print("- ESPACIO: pausar/reanudar")
        print("- +/-: acelerar/decelerar")
        print("- R: reiniciar")
        print("- 0-5: enfocar planetas (0=Sol, 1=Mercurio, etc.)")
        print("-" * 40)
        
        for step in range(steps):
            rate(30 * self.speed_factor)
            
            self.handle_input()

            if not self.paused:
                self.step(dt)
                
                # Actualizar enfoque si hay uno seleccionado
                if self.current_focus:
                    scene.center = vector(*self.current_focus.position)
                    self.viz.update(self.sim.bodies, focus_body=self.current_focus)
                else:
                    self.viz.update(self.sim.bodies)
                    
                self.total_time += dt
                # self.print_status()

    def calculate_total_energy(self):
        """Calcular energía total del sistema"""
        total_energy = 0
        G = 6.67430e-11
        
        # Energía cinética
        for body in self.sim.bodies:
            ke = 0.5 * body.mass * np.linalg.norm(body.velocity)**2
            total_energy += ke
        
        # Energía potencial
        for i, body1 in enumerate(self.sim.bodies):
            for body2 in self.sim.bodies[i+1:]:
                r = np.linalg.norm(body1.position - body2.position)
                pe = -G * body1.mass * body2.mass / r
                total_energy += pe
                
        return total_energy

if __name__ == "__main__":
    advanced_sim = AdvancedSimulation()
    advanced_sim.run()