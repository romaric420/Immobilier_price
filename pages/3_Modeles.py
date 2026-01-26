"""
Modèles ML - PriceWise
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Modèles - PriceWise", layout="wide")


# GÉNÉRATION DES DONNÉES


@st.cache_data
def load_data():
    np.random.seed(42)
    n = 3000
    
    cities = {
        'Paris': {'base': 10000, 'var': 2000},
        'Lyon': {'base': 4500, 'var': 800},
        'Marseille': {'base': 3200, 'var': 700},
        'Bordeaux': {'base': 4800, 'var': 900},
        'Lille': {'base': 3000, 'var': 600},
        'Nantes': {'base': 3800, 'var': 700},
        'Toulouse': {'base': 3500, 'var': 650},
        'Nice': {'base': 5500, 'var': 1200},
        'Strasbourg': {'base': 3300, 'var': 600},
        'Rennes': {'base': 3600, 'var': 650}
    }
    
    city_list = list(cities.keys())
    city_probs = [0.25, 0.12, 0.10, 0.10, 0.08, 0.08, 0.08, 0.07, 0.06, 0.06]
    
    data = {
        'property_id': [f'PROP_{i:05d}' for i in range(n)],
        'city': np.random.choice(city_list, n, p=city_probs),
        'surface': np.random.normal(70, 30, n).clip(15, 250).astype(int),
        'rooms': np.zeros(n, dtype=int),
        'construction_year': np.random.choice(range(1900, 2024), n),
        'property_type': np.random.choice(['Appartement', 'Maison', 'Studio', 'Loft'], n, p=[0.55, 0.30, 0.10, 0.05]),
        'energy_class': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n, p=[0.05, 0.10, 0.20, 0.25, 0.20, 0.12, 0.08]),
        'condition': np.random.choice(['Neuf', 'Très bon', 'Bon', 'À rénover'], n, p=[0.10, 0.30, 0.45, 0.15]),
        'has_parking': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'has_balcony': np.random.choice([0, 1], n, p=[0.5, 0.5]),
    }
    
    df = pd.DataFrame(data)
    df['rooms'] = (df['surface'] / 18).clip(1, 10).astype(int)
    
    base_prices = df['city'].map(lambda x: cities[x]['base'])
    variations = df['city'].map(lambda x: cities[x]['var'])
    
    price_per_m2 = (
        base_prices + 
        np.random.normal(0, 1, n) * variations +
        df['has_parking'] * 200 +
        df['has_balcony'] * 150 +
        df['energy_class'].map({'A': 500, 'B': 300, 'C': 100, 'D': 0, 'E': -100, 'F': -200, 'G': -400}) +
        df['condition'].map({'Neuf': 800, 'Très bon': 300, 'Bon': 0, 'À rénover': -500})
    ).clip(1500, 15000)
    
    df['price_per_m2'] = price_per_m2.round(0).astype(int)
    df['price'] = (df['price_per_m2'] * df['surface']).astype(int)
    
    return df

@st.cache_resource
def train_models(_X_train, _y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(_X_train)
    
    models = {
        'Régression Linéaire': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
        'KNN (k=10)': KNeighborsRegressor(n_neighbors=10)
    }
    
    trained = {}
    for name, model in models.items():
        model.fit(X_train_scaled, _y_train)
        trained[name] = model
    
    return trained, scaler


# PAGE


st.title("🤖 Modèles de Machine Learning")
st.markdown("Entraînement et évaluation des modèles de régression")

df = load_data()

# Préparation
df_ml = df.copy()
categorical_cols = ['city', 'property_type', 'energy_class', 'condition']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_ml[col + '_encoded'] = le.fit_transform(df_ml[col])
    label_encoders[col] = le

feature_cols = [
    'surface', 'rooms', 'construction_year', 'has_parking', 'has_balcony',
    'city_encoded', 'property_type_encoded', 'energy_class_encoded', 'condition_encoded'
]

X = df_ml[feature_cols]
y = df_ml['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models, scaler = train_models(X_train, y_train)

# Sidebar
st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox("Modèle à analyser", list(models.keys()))

st.markdown("---")

# Résultats
st.header("Comparaison des Performances")

X_test_scaled = scaler.transform(X_test)

results = []
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    results.append({
        'Modèle': name,
        'R²': f"{r2_score(y_test, y_pred):.4f}",
        'MAE': f"{mean_absolute_error(y_test, y_pred):,.0f} €",
        'RMSE': f"{np.sqrt(mean_squared_error(y_test, y_pred)):,.0f} €"
    })

st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

st.markdown("---")

# Graphiques
col1, col2 = st.columns(2)

with col1:
    # R² comparison
    r2_scores = [r2_score(y_test, models[name].predict(X_test_scaled)) for name in models.keys()]
    
    fig = go.Figure(go.Bar(
        x=list(models.keys()),
        y=r2_scores,
        marker_color='#2E86AB',
        text=[f"{s:.3f}" for s in r2_scores],
        textposition='outside'
    ))
    fig.update_layout(title="Comparaison R² Score", yaxis_title="R²", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Prédictions vs Réel
    y_pred = models[selected_model].predict(X_test_scaled)
    
    sample_idx = np.random.choice(len(y_test), min(500, len(y_test)), replace=False)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.array(y_test)[sample_idx] / 1000,
        y=y_pred[sample_idx] / 1000,
        mode='markers',
        marker=dict(color='#2E86AB', opacity=0.5),
        name='Prédictions'
    ))
    fig.add_trace(go.Scatter(
        x=[0, max(y_test) / 1000],
        y=[0, max(y_test) / 1000],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Parfait'
    ))
    fig.update_layout(title=f"Prédictions vs Réel - {selected_model}", 
                     xaxis_title="Prix Réel (k€)", yaxis_title="Prix Prédit (k€)", height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Validation croisée
st.header("Validation Croisée (5-Fold)")

X_scaled = scaler.transform(X)
cv_results = []

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    cv_results.append({
        'Modèle': name,
        'R² Moyen': f"{scores.mean():.4f}",
        'Écart-type': f"{scores.std():.4f}"
    })

st.dataframe(pd.DataFrame(cv_results), use_container_width=True, hide_index=True)
