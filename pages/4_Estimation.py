"""
Estimation - PriceWise
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor

st.set_page_config(page_title="Estimation - PriceWise", layout="wide")


# GÉNÉRATION DES DONNÉES ET MODÈLES


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
def prepare_and_train():
    df = load_data()
    df_ml = df.copy()
    
    label_encoders = {}
    categorical_cols = ['city', 'property_type', 'energy_class', 'condition']
    
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
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    models = {
        'Régression Linéaire': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
        'KNN (k=10)': KNeighborsRegressor(n_neighbors=10)
    }
    
    for model in models.values():
        model.fit(X_train_scaled, y_train)
    
    return df, models, scaler, label_encoders


# PAGE


st.title("Estimation de Bien")
st.markdown("Estimez le prix de votre bien immobilier")

df, models, scaler, label_encoders = prepare_and_train()

# Sidebar
st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox("Modèle", list(models.keys()))

st.markdown("---")

# Formulaire
st.header("Caractéristiques du Bien")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Localisation")
    city = st.selectbox("Ville", sorted(df['city'].unique()))
    property_type = st.selectbox("Type de bien", ['Appartement', 'Maison', 'Studio', 'Loft'])

with col2:
    st.subheader("Caractéristiques")
    surface = st.number_input("Surface (m²)", 15, 300, 70)
    rooms = st.number_input("Nombre de pièces", 1, 10, 3)
    construction_year = st.number_input("Année de construction", 1900, 2024, 1990)

with col3:
    st.subheader("Équipements")
    energy_class = st.selectbox("Classe DPE", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    condition = st.selectbox("État du bien", ['Neuf', 'Très bon', 'Bon', 'À rénover'])
    has_parking = st.checkbox("Parking")
    has_balcony = st.checkbox("Balcon/Terrasse")

st.markdown("---")

# Estimation
if st.button("Estimer le Prix", type="primary", use_container_width=True):
    
    # Préparation
    new_data = pd.DataFrame({
        'surface': [surface],
        'rooms': [rooms],
        'construction_year': [construction_year],
        'has_parking': [1 if has_parking else 0],
        'has_balcony': [1 if has_balcony else 0],
        'city_encoded': [label_encoders['city'].transform([city])[0]],
        'property_type_encoded': [label_encoders['property_type'].transform([property_type])[0]],
        'energy_class_encoded': [label_encoders['energy_class'].transform([energy_class])[0]],
        'condition_encoded': [label_encoders['condition'].transform([condition])[0]]
    })
    
    features_scaled = scaler.transform(new_data)
    
    # Prédictions de tous les modèles
    predictions = {name: model.predict(features_scaled)[0] for name, model in models.items()}
    avg_prediction = np.mean(list(predictions.values()))
    
    st.markdown("---")
    
    # Résultat principal
    st.header("Estimation")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #2E86AB 0%, #1E5A7E 100%); 
                    color: white; padding: 2rem; border-radius: 15px; text-align: center;">
            <h2 style="margin: 0; font-size: 2.5rem;">{avg_prediction:,.0f} €</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Estimation moyenne</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Prix au m² estimé:** {avg_prediction/surface:,.0f} €/m²")
    
    st.markdown("---")
    
    # Détails par modèle
    st.subheader("Estimations par Modèle")
    
    pred_df = pd.DataFrame({
        'Modèle': list(predictions.keys()),
        'Estimation': [f"{p:,.0f} €" for p in predictions.values()],
        'Prix/m²': [f"{p/surface:,.0f} €" for p in predictions.values()]
    })
    
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Comparaison marché
    st.subheader("Comparaison avec le Marché")
    
    similar = df[
        (df['city'] == city) &
        (df['property_type'] == property_type) &
        (df['surface'].between(surface * 0.8, surface * 1.2))
    ]
    
    if len(similar) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prix Moyen Similaires", f"{similar['price'].mean():,.0f} €")
        
        with col2:
            st.metric("Prix Min", f"{similar['price'].min():,.0f} €")
        
        with col3:
            st.metric("Prix Max", f"{similar['price'].max():,.0f} €")
        
        st.info(f"Comparaison basée sur {len(similar)} biens similaires à {city}.")
    else:
        st.warning("Pas assez de biens similaires pour comparaison.")
