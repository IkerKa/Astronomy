from vpython import sphere, vector, rate, canvas, color, label, distant_light, local_light, ring
import numpy as np

#TODO: meterle Z :3

class Visualizer:

    def __init__(self, bodies):
        # Canvas más grande y con fondo negro espacial
        self.canvas = canvas(
            title='🌌 Sistema Solar - Simulación N-Body', 
            width=1200, height=800,
            background=color.black,
            ambient=color.gray(0.2)  # Iluminación ambiental suave
            
        )
        
        # Añadir luz del Sol
        self.sun_light = local_light(pos=vector(0,0,0), color=color.white)
        
        self.canvas.autoscale = True
        self.spheres = {}
        self.show_labels = False
        self.labels = {}
        self.orbit_guides = []
        self.show_orbits = False
        

        for body in bodies:
            ball = sphere(
                pos=vector(body.position[0], body.position[1], body.position[2]),
                radius=self.get_radius(body),
                color=self.get_color(body),
                make_trail=True,
                trail_type="curve",
                retain=2000,  
                emissive=self.is_sun(body)  # El Sol brilla
                , trail_radius=self.get_radius(body) * 0.8
            )
            self.spheres[body] = ball

            self.create_starfield(30)  # Crear un campo de estrellas de fondo

            # Labels mejorados
            body_label = label(
                pos=vector(body.position[0], body.position[1], body.position[2]),
                text=body.name,
                color=color.white,
                background=color.gray(0.1),
                opacity=0.8,
                xoffset=25,
                yoffset=25,
                space=35,
                height=14,
                border=2,
                font='monospace'
            )
            self.labels[body] = body_label

        self.add_orbital_guides(bodies)

        self.particle_systems = {}
        for body in bodies:
            self.particle_systems[body] = []

        self.corona_effects = {}
        # Crear corona solar
        self.create_solar_corona()
        


    def create_solar_corona(self):
        """Crear efecto de corona solar"""
        corona_rings = []
        for i in range(5):
            corona = sphere(
                pos=vector(0, 0, 0),
                radius=8e9 * (1.2 + i * 0.3),
                color=color.yellow,
                opacity=0.05 - i * 0.008,
                emissive=True
            )
            corona_rings.append(corona)
        self.corona_effects['sol'] = corona_rings


    def create_comet_trail(self, body, intensity=1.0):
        """Crear efecto de cola de cometa"""
        if hasattr(body, 'is_comet') and body.is_comet:
            # Partículas brillantes que siguen al cometa
            trail_length = 20
            for i in range(trail_length):
                offset = vector(
                    -body.velocity[0] * i * 1e6,
                    -body.velocity[1] * i * 1e6,
                    0
                ) * intensity
                
                particle = sphere(
                    pos=vector(body.position[0], body.position[1], 0) + offset,
                    radius=self.get_radius(body) * (1 - i/trail_length) * 0.5,
                    color=color.white,
                    opacity=0.8 * (1 - i/trail_length),
                    emissive=True
                )

    def update_visual_effects(self, bodies):
        """Actualizar todos los efectos visuales"""
        for body in bodies:
            # Actualizar atmósferas
            if body in self.particle_systems:
                self.particle_systems[body].pos = vector(body.position[0], body.position[1], 0)
            
            # Actualizar vectores de velocidad
            if self.show_velocity_vectors and body in self.velocity_arrows:
                start_pos = vector(body.position[0], body.position[1], 0)
                vel_scale = 1e-4  # Factor de escala para visualizar velocidad
                end_pos = start_pos + vector(body.velocity[0], body.velocity[1], 0) * vel_scale
                
                # Actualizar la curva del vector
                self.velocity_arrows[body].clear()
                self.velocity_arrows[body].append(start_pos)
                self.velocity_arrows[body].append(end_pos)
            
            # Actualizar corona solar
            if self.is_sun(body) and 'sol' in self.corona_effects:
                sun_pos = vector(body.position[0], body.position[1], 0)
                for corona in self.corona_effects['sol']:
                    corona.pos = sun_pos

    
    def create_starfield(self, num_stars):
        """Crear un campo de estrellas de fondo para un efecto más realista"""
        for _ in range(num_stars):
            # Posición aleatoria en una esfera grande
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            r = 2e12  # Radio grande para las estrellas de fondo
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            # Tamaño y brillo aleatorios
            size = np.random.uniform(5e8, 2e9)
            brightness = np.random.uniform(0.3, 0.9)
            
            sphere(
                pos=vector(x, y, z),
                radius=size,
                color=color.white,
                opacity=brightness,
                emissive=True
            )

    def add_orbital_guides(self, bodies):
        """Añadir guías de órbitas circulares teóricas"""
        for body in bodies:
            if not self.is_sun(body):  
                orbital_radius = np.linalg.norm(body.position)
                
                orbit_guide = ring(
                    pos=vector(0, 0, 0),  
                    axis=vector(0, 0, 1),  
                    radius=orbital_radius,
                    thickness=orbital_radius * 0.001,  # Más delgado (era 0.002)
                    color=self.get_color(body),
                    opacity=0.2, 
                    visible=self.show_orbits
                )
                self.orbit_guides.append(orbit_guide)

    # Agregar esta función simple para alternar órbitas
    def toggle_orbit_guides(self):
        """Alternar visibilidad de las guías orbitales"""
        self.show_orbits = not self.show_orbits
        for guide in self.orbit_guides:
            guide.visible = self.show_orbits

    def is_sun(self, body):
        return "Sol" in body.name
    
    def get_radius(self, body):
        if hasattr(body, 'radius'):
            radius_value = body.radius[0] if isinstance(body.radius, (tuple, list)) else body.radius
        else:
            radius_value = 6.4e6  # Default Earth radius
        
        return radius_value * 5
        
    def get_color(self, body):
        """
        Si body tiene color definido, lo convierte a VPython color.
        """
        if hasattr(body, 'color'):
            return getattr(color, body.color.lower(), color.white)
    

    def update(self, bodies, focus_body = None):
        for body in bodies:
            if body in self.spheres:
                new_pos = vector(body.position[0], body.position[1], body.position[2])
                self.spheres[body].pos = new_pos
                
                # Actualizar luz del Sol
                if self.is_sun(body):
                    self.sun_light.pos = new_pos

                # Labels con más información
                speed = np.linalg.norm(body.velocity) / 1000
                distance = np.linalg.norm(body.position) / 1.496e11
                
                if self.show_labels:
                    self.labels[body].pos = new_pos
                    self.labels[body].text = f"🪐 {body.name}\n🚀 {speed:.1f} km/s\n📏 {distance:.2f} UA"

        if focus_body:
            self.canvas.center = vector(*focus_body.position) * 1.0



    def toggle_labels(self):
        for label_obj in self.labels.values():
            label_obj.visible = not label_obj.visible
