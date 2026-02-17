
import numpy as np


### Clase que representa una region cuadrada del espacio
class Quad:
    def __init__(self, x_center, y_center, z_center, length):
        self.x = x_center
        self.y = y_center
        self.z = z_center
        self.length = length        # lado del cuadrado

        
    def contains(self, position):
        """Verificar si una posición está dentro del cubo"""
        half = self.length / 2
        return (self.x - half <= position[0] < self.x + half and
                self.y - half <= position[1] < self.y + half and
                self.z - half <= position[2] < self.z + half)


    # 8 regiones/octantes en 3D
    def NWU(self):
        """Noroeste Superior"""
        return Quad(self.x - self.length/4, self.y + self.length/4, self.z + self.length/4, self.length/2)

    def NEU(self):
        """Noreste Superior"""
        return Quad(self.x + self.length/4, self.y + self.length/4, self.z + self.length/4, self.length/2)

    def SWU(self):
        """Suroeste Superior"""
        return Quad(self.x - self.length/4, self.y - self.length/4, self.z + self.length/4, self.length/2)

    def SEU(self):
        """Sureste Superior"""
        return Quad(self.x + self.length/4, self.y - self.length/4, self.z + self.length/4, self.length/2)

    def NWD(self):
        """Noroeste Inferior"""
        return Quad(self.x - self.length/4, self.y + self.length/4, self.z - self.length/4, self.length/2)

    def NED(self):
        """Noreste Inferior"""
        return Quad(self.x + self.length/4, self.y + self.length/4, self.z - self.length/4, self.length/2)

    def SWD(self):
        """Suroeste Inferior"""
        return Quad(self.x - self.length/4, self.y - self.length/4, self.z - self.length/4, self.length/2)

    def SED(self):
        """Sureste Inferior"""
        return Quad(self.x + self.length/4, self.y - self.length/4, self.z - self.length/4, self.length/2)

    def __repr__(self):
        return f"Quad(x={self.x}, y={self.y}, z={self.z}, length={self.length})"

## Clase que representa un nodo del QuadTree
class QuadTreeNode:
    def __init__(self, region):
        self.region = region       # instancia de Quad
        self.body = None           # si es hoja con un cuerpo
        self.children = {}         # {'NW': node, ...}
        self.mass = 0.0
        self.com = np.array([0.0, 0.0, 0.0])  # Usar numpy array en lugar de lista

    def is_external(self):
        return len(self.children) == 0
    
