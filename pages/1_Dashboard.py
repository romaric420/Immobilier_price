"""
Dashboard - PriceWise
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Dashboard - PriceWise", layout="wide")


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
    
    # Calcul du prix
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


# PAGE


st.title("Dashboard Marché Immobilier")
st.markdown("Vue d'ensemble des prix et tendances du marché français")

df = load_data()

# Filtres sidebar
st.sidebar.header("Filtres")

selected_cities = st.sidebar.multiselect(
    "Villes",
    df['city'].unique(),
    default=list(df['city'].unique())
)

price_range = st.sidebar.slider(
    "Fourchette de prix (€)",
    int(df['price'].min()),
    int(df['price'].max()),
    (int(df['price'].min()), int(df['price'].max()))
)

# Filtrage
df_filtered = df[
    (df['city'].isin(selected_cities)) &
    (df['price'] >= price_range[0]) &
    (df['price'] <= price_range[1])
]

# KPIs
st.header("Indicateurs Clés")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Biens Analysés", f"{len(df_filtered):,}")

with col2:
    st.metric("Prix Moyen", f"{df_filtered['price'].mean():,.0f} €")

with col3:
    st.metric("Prix Médian", f"{df_filtered['price'].median():,.0f} €")

with col4:
    st.metric("Prix/m² Moyen", f"{df_filtered['price_per_m2'].mean():,.0f} €/m²")

st.markdown("---")

# Graphiques
col1, col2 = st.columns(2)

with col1:
    # Distribution des prix
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_filtered['price'] / 1000, nbinsx=50, marker_color='#2E86AB'))
    fig.update_layout(title="Distribution des Prix", xaxis_title="Prix (k€)", yaxis_title="Nombre", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Prix par ville
    city_prices = df_filtered.groupby('city')['price_per_m2'].mean().sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=city_prices.values, y=city_prices.index, orientation='h', marker_color='#2E86AB'))
    fig.update_layout(title="Prix Moyen au m² par Ville", xaxis_title="€/m²", height=400)
    st.plotly_chart(fig, use_container_width=True)

# Scatter plot
st.header("Surface vs Prix")

fig = px.scatter(
    df_filtered.sample(min(1000, len(df_filtered))),
    x='surface',
    y='price',
    color='city',
    hover_data=['rooms', 'property_type'],
    opacity=0.6
)
fig.update_layout(xaxis_title="Surface (m²)", yaxis_title="Prix (€)", height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Tableau stats par ville
st.header("Statistiques par Ville")

city_stats = df_filtered.groupby('city').agg({
    'price': ['count', 'mean', 'median'],
    'price_per_m2': 'mean',
    'surface': 'mean'
}).round(0)

city_stats.columns = ['Nb Biens', 'Prix Moyen', 'Prix Médian', 'Prix/m²', 'Surface Moy.']
city_stats = city_stats.sort_values('Prix/m²', ascending=False)

st.dataframe(city_stats, use_container_width=True)
