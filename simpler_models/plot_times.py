import pandas as pd
import matplotlib.pyplot as plt

# Nombres de los métodos y archivos
metodos = ['leapfrog', 'euler', 'verlet', 'rk4']
archivos = [f'tiempo_{m}.csv' for m in metodos]
colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
estilos = ['o-', 's--', '^-', 'd:']

plt.figure(figsize=(10, 7))

for metodo, archivo, color, estilo in zip(metodos, archivos, colores, estilos):
    df = pd.read_csv(archivo, encoding='latin1')
    x = df[df.columns[0]]
    y = df['Tiempo (s)']
    plt.plot(x, y, estilo, color=color, label=metodo.capitalize(), markersize=7, linewidth=2)
    # Anotación del valor máximo
    max_idx = y.idxmax()
    plt.annotate(f'{y[max_idx]:.2f}s', (x[max_idx], y[max_idx]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color=color)

plt.xlabel('Años', fontsize=13)
plt.ylabel('Tiempo (s)', fontsize=13)
plt.title('Comparación de tiempos de integración', fontsize=15)
plt.xscale('log')
plt.legend(title='Método', fontsize=11)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('comparacion_tiempos.png', dpi=300)
plt.show()