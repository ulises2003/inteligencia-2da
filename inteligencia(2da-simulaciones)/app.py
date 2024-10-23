# app.py
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Cargar y preparar los datos de Iris
iris_data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv', header=None)
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

class MultiArmedBandit:
    def __init__(self):
        # Probabilidades de éxito para cada máquina
        self.probabilidades = [0.2, 0.5, 0.8]
        self.recompensas = [0] * len(self.probabilidades)
        self.tiradas = [0] * len(self.probabilidades)
        self.historial = []  # Nuevo: guardamos historial de resultados
    
    def jugar_brazo(self, brazo):
        resultado = 1 if np.random.random() < self.probabilidades[brazo] else 0
        self.recompensas[brazo] += resultado
        self.tiradas[brazo] += 1
        
        # Guardar en historial
        self.historial.append({
            'brazo': brazo,
            'resultado': resultado,
            'timestamp': pd.Timestamp.now()
        })
        
        return {
            'resultado': resultado,
            'recompensas': self.recompensas,
            'tiradas': self.tiradas,
            'tasas': [
                round(self.recompensas[i] / self.tiradas[i] * 100, 2) 
                if self.tiradas[i] > 0 else 0 
                for i in range(len(self.probabilidades))
            ],
            'ultimas_jugadas': self.obtener_ultimas_jugadas(10)
        }
    
    def obtener_ultimas_jugadas(self, n):
        return [
            {
                'brazo': h['brazo'],
                'resultado': h['resultado'],
                'tiempo': h['timestamp'].strftime('%H:%M:%S')
            }
            for h in self.historial[-n:][::-1]
        ]

# Instancia global del Multi-Armed Bandit
bandido = MultiArmedBandit()

def generar_grafico_pca(feature1, feature2):
    # Seleccionar características y normalizar
    features = iris_data[[feature1, feature2]]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Aplicar PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(features_scaled)
    
    # Calcular varianza explicada
    var_ratio = pca.explained_variance_ratio_
    var_total = sum(var_ratio)
    
    # Crear DataFrame con componentes principales
    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    pca_df['species'] = iris_data['species']
    
    # Crear gráfico
    plt.figure(figsize=(12, 8))
    
    # Plot principal
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='species', 
                   palette='viridis', s=100, alpha=0.7)
    plt.title(f'PCA de Iris: {feature1} vs {feature2}')
    plt.xlabel(f'PC1 ({var_ratio[0]*100:.1f}% varianza)')
    plt.ylabel(f'PC2 ({var_ratio[1]*100:.1f}% varianza)')
    
    # Añadir subplot con biplot
    plt.subplot(1, 2, 2)
    loadings = pca.components_.T
    for i, feature in enumerate([feature1, feature2]):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                 color='r', alpha=0.5, head_width=0.05)
        plt.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, feature)
    
    plt.scatter(components[:, 0], components[:, 1], c='gray', alpha=0.2)
    plt.title('Biplot: Dirección de Features')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    plt.tight_layout()
    
    # Convertir gráfico a imagen base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return {
        'plot_url': f'data:image/png;base64,{plot_url}',
        'var_explicada': {
            'pc1': round(var_ratio[0] * 100, 2),
            'pc2': round(var_ratio[1] * 100, 2),
            'total': round(var_total * 100, 2)
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pca', methods=['GET', 'POST'])
def pca():
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    result = None
    
    if request.method == 'POST':
        feature1 = request.form.get('feature1')
        feature2 = request.form.get('feature2')
        if feature1 and feature2 and feature1 != feature2:
            result = generar_grafico_pca(feature1, feature2)
    
    return render_template('pca.html', 
                         features=features, 
                         result=result)

@app.route('/bandit')
def bandit():
    return render_template('bandit.html')

@app.route('/tirar/<int:brazo>')
def tirar(brazo):
    if 0 <= brazo < len(bandido.probabilidades):
        return jsonify(bandido.jugar_brazo(brazo))
    return jsonify({'error': 'Brazo inválido'}), 400

if __name__ == '__main__':
    app.run(debug=True)