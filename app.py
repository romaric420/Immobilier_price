"""
PriceWise - Estimation Immobilière
Page d'accueil
"""

import streamlit as st

st.set_page_config(
    page_title="PriceWise",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin-bottom: 1rem;
    }
    .tech-badge {
        background: #2E86AB;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">PriceWise</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Estimation Immobilière Intelligente par Machine Learning</p>', unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("À propos du projet")
    
    st.markdown("""
    **PriceWise** est une application de Machine Learning conçue pour estimer 
    le prix des biens immobiliers sur le marché français. Elle compare plusieurs 
    algorithmes de régression pour fournir des estimations précises.
    
    ### Problématique Business
    
    L'estimation immobilière est un enjeu crucial pour :
    
    - Les acheteurs qui veulent évaluer un bien
    - Les vendeurs qui souhaitent fixer le bon prix
    - Les agences immobilières pour leurs mandats
    - Les banques pour l'évaluation des garanties
    """)
    
    st.header("Fonctionnalités")
    
    st.markdown("""
    <div class="feature-card">
        <strong>Dashboard</strong><br>
        Vue d'ensemble des prix par ville et type de bien
    </div>
    
    <div class="feature-card">
        <strong>Analyse</strong><br>
        Facteurs influençant les prix : DPE, état, surface, localisation
    </div>
    
    <div class="feature-card">
        <strong>Modèles</strong><br>
        5 algorithmes de régression avec métriques détaillées
    </div>
    
    <div class="feature-card">
        <strong>Estimation</strong><br>
        Prédiction de prix pour un bien spécifique
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.header("Technologies")
    
    st.markdown("""
    <span class="tech-badge">Python</span>
    <span class="tech-badge">Streamlit</span>
    <span class="tech-badge">Scikit-learn</span>
    <span class="tech-badge">Pandas</span>
    <span class="tech-badge">Plotly</span>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("Modèles")
    
    st.markdown("""
    | Modèle | Type |
    |--------|------|
    | Régression Linéaire | Baseline |
    | Ridge | Régularisé L2 |
    | Lasso | Régularisé L1 |
    | KNN (k=5) | Non-paramétrique |
    | KNN (k=10) | Non-paramétrique |
    """)
    
    st.markdown("---")
    
    st.header("Métriques")
    
    st.markdown("""
    - R² Score
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    """)

st.markdown("---")

st.header("Navigation")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("**1 - Dashboard**\n\nVue du marché")

with col2:
    st.info("**2 - Analyse**\n\nFacteurs de prix")

with col3:
    st.info("**3 - Modèles**\n\nPerformance ML")

with col4:
    st.info("**4 - Estimation**\n\nPrédiction de prix")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "Utilisez le menu latéral pour naviguer"
    "</div>",
    unsafe_allow_html=True
)
