# app.py
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Configuraciones
DATASET_PATH = 'ferreteria_dataset_large.csv'
STATIC_PATH = os.path.join('static', 'images')
os.makedirs(STATIC_PATH, exist_ok=True)

# También necesitamos modificar la clase MultiArmedBandit para usar tipos nativos
class MultiArmedBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_reward = 0.0
        self.history = []
    
    def select_arm(self, epsilon=0.1):
        if np.random.random() < epsilon:
            return int(np.random.randint(self.n_arms))
        return int(np.argmax(self.values))
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
        self.total_reward += reward
        self.history.append({
            'arm': int(chosen_arm),
            'reward': float(reward),
            'cumulative_reward': float(self.total_reward)
        })

def load_and_preprocess_data():
    df = pd.read_csv(DATASET_PATH)
    df['Genero'] = df['Genero'].map({'F': 0, 'M': 1})
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pca')
def pca_analysis():
    # Cargar y preprocesar datos
    df = load_and_preprocess_data()
    features = ['Edad', 'Genero', 'Precio', 'Cantidad', 
               'Dias_Desde_Ultima_Compra', 'Total_Compras', 'Descuento_Aplicado']
    X = df[features]
    
    # Estandarizar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA
    pca = PCA()
    principal_components = pca.fit_transform(X_scaled)
    
    # Crear visualizaciones
    # 1. Scatter plot de los dos primeros componentes
    plt.figure(figsize=(10, 6))
    plt.clf()
    scatter_data = pd.DataFrame(data=principal_components[:, :2], 
                              columns=['PC1', 'PC2'])
    scatter_data['Compra_Futura'] = df['Compra_Futura']
    sns.scatterplot(data=scatter_data, x='PC1', y='PC2', 
                   hue='Compra_Futura', palette='viridis')
    plt.title('Análisis de Componentes Principales')
    plt.savefig(os.path.join(STATIC_PATH, 'pca_scatter.png'))
    
    # 2. Gráfico de varianza explicada
    plt.figure(figsize=(10, 6))
    plt.clf()
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    plt.plot(range(1, len(explained_variance_ratio) + 1), 
            cumulative_variance_ratio, 'bo-')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title('Varianza Explicada vs Número de Componentes')
    plt.grid(True)
    plt.savefig(os.path.join(STATIC_PATH, 'variance_explained.png'))
    
    # Preparar datos para la template
    component_weights = pd.DataFrame(
        pca.components_[:2, :],
        columns=features,
        index=['PC1', 'PC2']
    ).round(3).to_dict('index')
    
    explained_variance = {
        'individual': explained_variance_ratio.round(3).tolist(),
        'cumulative': cumulative_variance_ratio.round(3).tolist()
    }
    
    return render_template(
        'pca.html',
        scatter_plot='images/pca_scatter.png',
        variance_plot='images/variance_explained.png',
        component_weights=component_weights,
        explained_variance=explained_variance
    )

@app.route('/bandit')
def bandit_analysis():
    # Cargar datos
    df = load_and_preprocess_data()
    
    # Simular diferentes estrategias de descuento como brazos del bandit
    descuentos = [0, 0.05, 0.10, 0.15, 0.20]
    n_simulaciones = 1000
    
    # Inicializar bandit
    bandit = MultiArmedBandit(len(descuentos))
    
    # Simular decisiones y recompensas
    for _ in range(n_simulaciones):
        arm = bandit.select_arm()
        descuento = descuentos[arm]
        
        similar_purchases = df[
            (df['Descuento_Aplicado'] >= descuento - 0.02) & 
            (df['Descuento_Aplicado'] <= descuento + 0.02)
        ]
        
        if len(similar_purchases) > 0:
            reward = float(similar_purchases['Compra_Futura'].mean())  # Convertir a float
        else:
            reward = 0.0
            
        bandit.update(arm, reward)
    
    # Convertir arrays numpy a listas Python y valores numpy a tipos Python nativos
    results = {
        'descuentos': descuentos,
        'valores_estimados': [float(v) for v in bandit.values],  # Convertir a float
        'veces_seleccionado': [int(c) for c in bandit.counts],   # Convertir a int
        'mejor_descuento': float(descuentos[np.argmax(bandit.values)]),  # Convertir a float
        'history': [
            {
                'arm': int(h['arm']),  # Convertir a int
                'reward': float(h['reward']),  # Convertir a float
                'cumulative_reward': float(h['cumulative_reward'])  # Convertir a float
            }
            for h in bandit.history[-100:]  # Últimas 100 iteraciones
        ]
    }
    
    return render_template('bandit.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)