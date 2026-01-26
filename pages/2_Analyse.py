"""
Analyse Exploratoire - PriceWise
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Analyse - PriceWise", layout="wide")


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


# PAGE


st.title("Analyse Exploratoire")
st.markdown("Exploration des facteurs influençant les prix immobiliers")

df = load_data()

st.markdown("---")

# Analyse par DPE
st.header("Impact du DPE sur les Prix")

col1, col2 = st.columns(2)

with col1:
    energy_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    energy_prices = df.groupby('energy_class')['price_per_m2'].mean().reindex(energy_order)
    
    fig = go.Figure(go.Bar(
        x=energy_order,
        y=energy_prices.values,
        marker_color=['#2ECC71', '#27AE60', '#F1C40F', '#E67E22', '#E74C3C', '#C0392B', '#8E44AD']
    ))
    fig.update_layout(title="Prix/m² par Classe Énergétique", yaxis_title="€/m²", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    condition_prices = df.groupby('condition')['price_per_m2'].mean().sort_values(ascending=True)
    
    fig = go.Figure(go.Bar(
        x=condition_prices.values,
        y=condition_prices.index,
        orientation='h',
        marker_color='#3498DB'
    ))
    fig.update_layout(title="Prix/m² par État du Bien", xaxis_title="€/m²", height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Analyse par type de bien
st.header("Analyse par Type de Bien")

col1, col2 = st.columns(2)

with col1:
    type_prices = df.groupby('property_type')['price'].mean().sort_values(ascending=True)
    
    fig = go.Figure(go.Bar(
        x=type_prices.values,
        y=type_prices.index,
        orientation='h',
        marker_color='#E94F37'
    ))
    fig.update_layout(title="Prix Moyen par Type", xaxis_title="€", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    type_surface = df.groupby('property_type')['surface'].mean().sort_values(ascending=True)
    
    fig = go.Figure(go.Bar(
        x=type_surface.values,
        y=type_surface.index,
        orientation='h',
        marker_color='#F39C12'
    ))
    fig.update_layout(title="Surface Moyenne par Type", xaxis_title="m²", height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Matrice de corrélation
st.header("Matrice de Corrélation")

numeric_cols = ['surface', 'rooms', 'construction_year', 'has_parking', 'has_balcony', 'price_per_m2', 'price']
corr_matrix = df[numeric_cols].corr()

fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',
    zmid=0,
    text=corr_matrix.values.round(2),
    texttemplate='%{text}'
))
fig.update_layout(title="Corrélation entre Variables", height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Insights
st.header("Insights Clés")

col1, col2, col3 = st.columns(3)

with col1:
    best_dpe = df[df['energy_class'].isin(['A', 'B'])]['price_per_m2'].mean()
    worst_dpe = df[df['energy_class'].isin(['F', 'G'])]['price_per_m2'].mean()
    diff_dpe = ((best_dpe / worst_dpe) - 1) * 100
    
    st.info(f"**Impact DPE**\n\nClasse A-B vs F-G\n\n+{diff_dpe:.0f}% de valeur")

with col2:
    neuf = df[df['condition'] == 'Neuf']['price_per_m2'].mean()
    renov = df[df['condition'] == 'À rénover']['price_per_m2'].mean()
    diff_etat = ((neuf / renov) - 1) * 100
    
    st.warning(f"**Impact État**\n\nNeuf vs À rénover\n\n+{diff_etat:.0f}% de valeur")

with col3:
    paris = df[df['city'] == 'Paris']['price_per_m2'].mean()
    province = df[df['city'] != 'Paris']['price_per_m2'].mean()
    diff_loc = ((paris / province) - 1) * 100
    
    st.success(f"**Impact Localisation**\n\nParis vs Province\n\n+{diff_loc:.0f}% de valeur")
