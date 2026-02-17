

### Diseño del cuerpo orbital

import numpy as np
import matplotlib.pyplot as plt

class Body:
    def __init__(self, name, mass, position, velocity, color = 'Blue', radius = 1):
        # La aceleracion la calcularemos en cada paso
        self.name = name
        self.mass = mass
        self.color = color
        self.position = np.array(position, dtype=float) #[x,y,z si se añade]
        self.radius = radius, # radio para visualización
        self.velocity = np.array(velocity, dtype=float) #[vx,vy,vz si se añade]

    def __repr__(self):
        return f"<Body {self.name}: mass={self.mass}, pos={self.position}, vel={self.velocity}>"
        