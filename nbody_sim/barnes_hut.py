
from quadtree import Quad, QuadTreeNode
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Lo hice en 2D pero lo dejo en 3D
class BarnesHutTree:
    def __init__(self, bodies, theta=0.5, G=6.67430e-11):
        self.region = self._calculate_bounds(bodies)
        self.root = QuadTreeNode(self.region)        # Region es un Quad
        self.theta = theta
        self.G = G

        for body in bodies:
            self.insert(self.root, body, depth=0)

    def insert(self, node, body, depth=0):
        # Si el nodo está vacío
        MAX_DEPTH = 20  # Profundidad máxima para evitar recursión infinita
        if node.body is None and node.is_external():
            node.body = body
            node.mass = body.mass
            node.com = body.position.copy()
            return

        # Si es interno, actualiza masa y COM, luego propaga
        if not node.is_external():
            old_mass = node.mass
            old_com = node.com.copy()
            node.mass += body.mass
            # Actualiza el centro de masa correctamente
            node.com = self._update_com(old_mass, old_com, body.mass, body.position)
            self._put_in_child(node, body, depth + 1)
            return

        # Si es externo y ya tiene un cuerpo → subdividir
        if node.body is not None and depth < MAX_DEPTH:
            old_body = node.body
            node.body = None
            self._subdivide(node)
            
            # Actualiza masa y centro de masa ANTES de insertar
            node.mass = old_body.mass + body.mass
            node.com = self._update_com(old_body.mass, old_body.position, body.mass, body.position)
            
            # Ahora inserta ambos cuerpos
            self._put_in_child(node, old_body, depth + 1)
            self._put_in_child(node, body, depth + 1)
        
        elif depth >= MAX_DEPTH:
            # Mantenemos ambos cuerpos en el nodo sin subdividir
            node.body = body
            node.mass += body.mass
            node.com = self._update_com(node.mass - body.mass, node.com, body.mass, body.position)

    def _update_com(self, m1, com1, m2, com2):
        """Actualiza el centro de masa dados dos masas y sus posiciones"""
        if m1 == 0:
            return np.array(com2)
        if m2 == 0:
            return np.array(com1)
        
        total_mass = m1 + m2
        return np.array([
            (m1 * com1[0] + m2 * com2[0]) / total_mass,
            (m1 * com1[1] + m2 * com2[1]) / total_mass,
            (m1 * com1[2] + m2 * com2[2]) / total_mass
        ])

    def _calculate_bounds(self, bodies):
        """
        Limites del espacio que contiene todos los cuerpos.
        """

        if not bodies:
            return Quad(0, 0, 0, 1)
        
        positions = np.array([body.position for body in bodies])
        
        # Encontrar límites
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        
        # Centro del bounding box
        center = (min_coords + max_coords) / 2
        
        # Tamaño de cada dimensión
        sizes = max_coords - min_coords
        
        # Usar el tamaño máximo para crear un cubo
        max_size = np.max(sizes)
        
        # Añadir margen de seguridad (20%)
        length = max_size * 1.2
        
        # Longitud mínima
        if length < 1.0:
            length = 1.0
        
        # Verificar que todos los puntos están dentro
        half_length = length / 2
        for i, body in enumerate(bodies):
            pos = body.position
            #Si la posición del cuerpo está fuera de los límites, imprimir advertencia
            if (abs(pos[0] - center[0]) > half_length or 
                abs(pos[1] - center[1]) > half_length or 
                abs(pos[2] - center[2]) > half_length):
                print(f"⚠️ Cuerpo {body.name} fuera de bounds: {pos} vs centro {center} ± {half_length}")
        
        return Quad(center[0], center[1], center[2], length)

    def _subdivide(self, node):
        q = node.region
        # Crear 8 hijos
        node.children['NWU'] = QuadTreeNode(q.NWU())
        node.children['NEU'] = QuadTreeNode(q.NEU())
        node.children['SWU'] = QuadTreeNode(q.SWU())
        node.children['SEU'] = QuadTreeNode(q.SEU())
        node.children['NWD'] = QuadTreeNode(q.NWD())
        node.children['NED'] = QuadTreeNode(q.NED())
        node.children['SWD'] = QuadTreeNode(q.SWD())
        node.children['SED'] = QuadTreeNode(q.SED())
        


    def _put_in_child(self, node, body, depth):
        for key, child in node.children.items():
            if child.region.contains(body.position):
                self.insert(child, body, depth)
                return  # Salir tras insertar en la subregión correcta
        # Si ninguna subregión contiene el cuerpo, aquí sí es error
        print(f"¡ERROR! El cuerpo {body.name} NO cabe en ninguna subregión de {node.region}")



    def compute_force(self, target_body):
        return self._compute_force_recursive(self.root, target_body)

    
    def _compute_force_recursive(self, node, target_body):
        if node is None or (node.is_external() and node.body == target_body):
            return np.array([0.0, 0.0, 0.0])

        # Calcular distancia
        dx = node.com[0] - target_body.position[0]
        dy = node.com[1] - target_body.position[1]
        dz = node.com[2] - target_body.position[2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        if dist < 1e-10:  # Evitar división por cero
            return np.array([0.0, 0.0, 0.0])

        s = node.region.length
        d = np.linalg.norm(node.com - target_body.position)
        if node.is_external() or (s / d) < self.theta:
            # print(f"Calculando fuerza para {target_body.name} desde nodo {node.region}")
            if node.mass == 0:
                return np.array([0.0, 0.0, 0.0])
            
            force_mag = self.G * target_body.mass * node.mass / dist**2  
            force_vec = force_mag * np.array([dx, dy, dz]) / dist    
            return force_vec
        else:
            total_force = np.array([0.0, 0.0, 0.0])
            for child in node.children.values():
                if child is not None:
                    total_force += self._compute_force_recursive(child, target_body)
            return total_force
        


    def draw_tree_simple(self, ax, max_depth=None):
        """Dibuja el árbol de forma simple y clara"""
        self._draw_node_simple(ax, self.root, 0, max_depth)
        
        # Configurar el plot
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

    def _draw_node_simple(self, ax, node, depth, max_depth):
        """Dibuja un nodo de forma simple"""
        if node is None:
            return
        
        # Limitar profundidad si se especifica
        if max_depth is not None and depth > max_depth:
            return
        
        colors = ['black', 'blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
        edge_color = colors[min(depth, len(colors)-1)]
        linewidth = max(0.3, 2.0 - depth * 0.3)  # Cuando más profundo, más delgado
        
        # Dibujar la región del nodo
        rect = Rectangle(
            (node.region.x - node.region.length / 2, 
            node.region.y - node.region.length / 2),
            node.region.length, node.region.length,
            fill=False, edgecolor=edge_color, linewidth=linewidth, alpha=0.7
        )
        ax.add_patch(rect)
        
        if node.is_external() and node.body is not None:
            ax.scatter(node.body.position[0], node.body.position[1], 
                    color=node.body.color, s=50, zorder=10, 
                    edgecolor='black', linewidth=1)
            
            ax.annotate(node.body.name, 
                    (node.body.position[0], node.body.position[1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold')
        
        # Dibujaremos una cruz roja solo en los nodos internos que tienen mas de un cuerpo
        elif not node.is_external() and hasattr(node, 'com') and node.mass > 0:
            ax.scatter(node.com[0], node.com[1], 
                    color='red', marker='x', s=60, zorder=8,
                    linewidth=2)
            
            # ax.annotate(f'{node.mass:.1e}', 
            #         (node.com[0], node.com[1]),
            #         xytext=(0, 10), textcoords='offset points',
            #         fontsize=7, ha='center', 
            #         bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
        

        if not node.is_external():
            for child in node.children.values():
                if child is not None:
                    self._draw_node_simple(ax, child, depth + 1, max_depth)

    def draw_tree_hierarchical(self, ax):
        """Dibuja el árbol resaltando la jerarquía"""
        levels = self._get_tree_levels()
        
        level_styles = [
            {'color': 'black', 'linewidth': 3, 'alpha': 1.0},      
            {'color': 'blue', 'linewidth': 2, 'alpha': 0.8},       
            {'color': 'green', 'linewidth': 1.5, 'alpha': 0.7},    
            {'color': 'red', 'linewidth': 1, 'alpha': 0.6},        
            {'color': 'purple', 'linewidth': 0.8, 'alpha': 0.5},   
        ]
        
        for level, nodes in levels.items():
            style = level_styles[min(level, len(level_styles)-1)]
            
            for node in nodes:
                rect = Rectangle(
                    (node.region.x - node.region.length / 2,
                    node.region.y - node.region.length / 2),
                    node.region.length, node.region.length,
                    fill=False, **style
                )
                ax.add_patch(rect)
        
        for node in self._traverse_nodes(self.root):
            if node.is_external() and node.body is not None:
                ax.scatter(node.body.position[0], node.body.position[1],
                        color=node.body.color, s=100, zorder=10,
                        edgecolor='white', linewidth=2)
                ax.annotate(node.body.name,
                        (node.body.position[0], node.body.position[1]),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', alpha=0.8))
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

    def draw_tree_focused(self, ax, focus_body_name=None):
        """Dibuja el árbol enfocándose en un cuerpo específico"""
        focus_node = None
        if focus_body_name:
            for node in self._traverse_nodes(self.root):
                if (node.is_external() and node.body is not None and 
                    node.body.name == focus_body_name):
                    focus_node = node
                    break
        
        if focus_node:
            path_to_focus = self._get_path_to_node(focus_node)
            
            for i, node in enumerate(path_to_focus):
                color = 'gold' if i == len(path_to_focus)-1 else 'orange'
                linewidth = 3 - i * 0.3
                
                rect = Rectangle(
                    (node.region.x - node.region.length / 2,
                    node.region.y - node.region.length / 2),
                    node.region.length, node.region.length,
                    fill=False, edgecolor=color, linewidth=linewidth
                )
                ax.add_patch(rect)
        
       
        for node in self._traverse_nodes(self.root):
            if focus_node is None or node not in path_to_focus:
                rect = Rectangle(
                    (node.region.x - node.region.length / 2,
                    node.region.y - node.region.length / 2),
                    node.region.length, node.region.length,
                    fill=False, edgecolor='lightgray', linewidth=0.5, alpha=0.5
                )
                ax.add_patch(rect)
        
        for node in self._traverse_nodes(self.root):
            if node.is_external() and node.body is not None:
                size = 100 if node == focus_node else 50
                ax.scatter(node.body.position[0], node.body.position[1],
                        color=node.body.color, s=size, zorder=10)
                ax.annotate(node.body.name,
                        (node.body.position[0], node.body.position[1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8 if node != focus_node else 12,
                        fontweight='bold' if node == focus_node else 'normal')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Árbol Barnes-Hut - Enfoque en {focus_body_name or "General"}')

    def _get_tree_levels(self):
        """Obtiene nodos organizados por nivel"""
        levels = {}
        
        def _classify_by_level(node, level):
            if node is None:
                return
            
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
            
            # Si no es externo, seguir con los hijos
            if not node.is_external():
                for child in node.children.values():
                    if child is not None:
                        _classify_by_level(child, level + 1)
        
        _classify_by_level(self.root, 0)
        return levels

    def _get_path_to_node(self, target_node):
        """Obtiene el camino desde la raíz hasta un nodo específico"""
        path = []
        
        def _find_path(node, target):
            if node is None:
                return False
            
            path.append(node)
            
            if node == target:
                return True
            
            # Si es interno, buscar en los hijos y si hay coincidencia, retornar True
            if not node.is_external():
                for child in node.children.values():
                    if child is not None and _find_path(child, target):
                        return True
            # Si no se encontró en los hijos, quitar el nodo actual del camino
            path.pop()
            return False
        
        _find_path(self.root, target_node)
        return path
    
    def draw_3D_simple(self, ax):
        """Dibuja el árbol en 3D de forma simple y clara"""

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')  
        ax.set_zlabel('z (m)')
        ax.set_title('Barnes-Hut Tree 3D')
        
        for node in self._traverse_nodes(self.root):
            self._draw_node_3d_simple(ax, node)

    def _draw_node_3d_simple(self, ax, node):
        """Dibuja un nodo individual en 3D"""
        if node is None:
            return
        
        region = node.region
        half_length = region.length / 2
        
        x_center, y_center, z_center = region.x, region.y, region.z
        
        corners = [
            [x_center - half_length, y_center - half_length, z_center - half_length],  # 0
            [x_center + half_length, y_center - half_length, z_center - half_length],  # 1
            [x_center + half_length, y_center + half_length, z_center - half_length],  # 2
            [x_center - half_length, y_center + half_length, z_center - half_length],  # 3
            [x_center - half_length, y_center - half_length, z_center + half_length],  # 4
            [x_center + half_length, y_center - half_length, z_center + half_length],  # 5
            [x_center + half_length, y_center + half_length, z_center + half_length],  # 6
            [x_center - half_length, y_center + half_length, z_center + half_length],  # 7
        ]
        
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  
            [4, 5], [5, 6], [6, 7], [7, 4],  
            [0, 4], [1, 5], [2, 6], [3, 7]   
        ]
        
        # El cubo sera azul si es externo y tiene cuerpo, gris claro si no tiene cuerpo
        if node.is_external():
            color = 'blue' if node.body is not None else 'lightgray'
            alpha = 0.8 if node.body is not None else 0.3
            linewidth = 1.5 if node.body is not None else 0.5
        else:
            color = 'red'
            alpha = 0.6
            linewidth = 1.0
        
        for edge in edges:
            start, end = corners[edge[0]], corners[edge[1]]
            ax.plot([start[0], end[0]], 
                    [start[1], end[1]], 
                    [start[2], end[2]], 
                    color=color, alpha=alpha, linewidth=linewidth)
        
        if node.is_external() and node.body is not None:
            ax.scatter(node.body.position[0], 
                    node.body.position[1], 
                    node.body.position[2],
                    color=node.body.color, s=100, zorder=10,
                    edgecolor='black', linewidth=1)
            
            
            ax.text(node.body.position[0], 
                    node.body.position[1], 
                    node.body.position[2] + region.length * 0.1,
                    node.body.name, fontsize=8, ha='center')
        
        elif not node.is_external() and hasattr(node, 'com') and node.mass > 0:
            ax.scatter(node.com[0], node.com[1], node.com[2],
                    color='red', marker='x', s=80, zorder=8, linewidth=2)
            
    def draw_minimalistic(self, ax):
        """
        Dibuja el árbol de arriba a abajo, cada nivel es una fila.
        Nodo con cuerpo: bola sólida. Nodo sin cuerpo: cuadrado con baja opacidad.
        """
        levels = self._get_tree_levels()
        
        if not levels:
            ax.text(0.5, 0.5, 'Árbol vacío', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=14)
            return
        
        y_gap = 3.0
        x_gap = 2.5
        
        node_positions = {}
        
        for level, nodes in levels.items():
            y = -level * y_gap
            n = len(nodes)
            
            # Ajustes únicamente visuales
            if n == 1:
                x_start = 0
            else:
                x_start = -(n - 1) * x_gap / 2
            
            for i, node in enumerate(nodes):
                x = x_start + i * x_gap
                node_positions[node] = (x, y)
        
        
        for level, nodes in levels.items():
            for node in nodes:
                if not node.is_external() and node in node_positions:
                    x_parent, y_parent = node_positions[node]
                    
                    for child in node.children.values():
                        if child is not None and child in node_positions:
                            x_child, y_child = node_positions[child]
                            ax.plot([x_parent, x_child], [y_parent, y_child], 
                                color='gray', linewidth=1.5, alpha=0.7, zorder=1)
        
        for node, (x, y) in node_positions.items():
            if node.is_external() and node.body is not None:
                ax.scatter(x, y, s=200, color=node.body.color, 
                        edgecolor='black', linewidth=2, zorder=10)
                
                ax.annotate(node.body.name, (x, y), 
                        xytext=(0, 25), textcoords='offset points',
                        fontsize=10, ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', alpha=0.8, edgecolor='black'))
            
            elif node.is_external() and node.body is None:
                # Menos opacidad si no hay cuerpo en una hoja
                rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, 
                            fill=True, color='lightgray', alpha=0.3, 
                            edgecolor='gray', linewidth=1, zorder=5)
                ax.add_patch(rect)
                
                ax.text(x, y, 'Ø', ha='center', va='center', 
                    fontsize=8, color='gray', alpha=0.7)
            
            else:
                mass_color = 'lightblue' if node.mass > 0 else 'lightgray'
                rect = Rectangle((x - 0.6, y - 0.6), 1.2, 1.2, 
                            fill=True, color=mass_color, alpha=0.6, 
                            edgecolor='black', linewidth=1.5, zorder=5)
                ax.add_patch(rect)
                
                
                if node.mass > 0:
                    ax.text(x, y + 0.15, f'{node.mass:.1e}', 
                        ha='center', va='center', fontsize=7, 
                        fontweight='bold', color='darkblue')
                    
                   
                    body_count = self._count_bodies_in_subtree(node)
                    ax.text(x, y - 0.15, f'({body_count})', 
                        ha='center', va='center', fontsize=6, 
                        color='darkgreen')
                else:
                    ax.text(x, y, 'ε', ha='center', va='center', 
                        fontsize=10, color='gray', alpha=0.7)
        
        #--Plot settings--
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Árbol Barnes-Hut - Vista Jerárquica', fontsize=18, fontweight='bold', pad=30)

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', 
                markersize=16, markeredgecolor='black', markeredgewidth=2, 
                label='Cuerpos físicos'),
            Patch(facecolor='lightblue', alpha=0.6, edgecolor='black', 
                label='Nodos internos (con masa)'),
            Patch(facecolor='lightgray', alpha=0.3, edgecolor='gray', 
                label='Nodos vacíos'),
            Line2D([0], [0], color='gray', linewidth=2.5, alpha=0.7, 
                label='Conexiones padre-hijo')
        ]

        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=14)

        # Añadir información del árbol
        total_nodes = len(node_positions)
        max_depth = max(levels.keys()) if levels else 0
        total_bodies = sum(1 for node in node_positions.keys() 
                        if node.is_external() and node.body is not None)

        info_text = (f"Nodos totales: {total_nodes}\n"
                    f"Profundidad: {max_depth}\n"
                    f"Cuerpos: {total_bodies}\n"
                    f"θ = {self.theta}")

        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
            fontsize=13, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='wheat', alpha=0.85))

    def _count_bodies_in_subtree(self, node):
        """Cuenta los cuerpos en el subárbol de un nodo"""
        if node.is_external():
            return 1 if node.body is not None else 0
        
        count = 0
        for child in node.children.values():
            if child is not None:
                count += self._count_bodies_in_subtree(child)
        return count
             
        
    def _traverse_nodes(self, node):
        """Generador para recorrer los nodos del árbol"""
        if node is None:
            return
        yield node
        for child in node.children.values():
            yield from self._traverse_nodes(child)