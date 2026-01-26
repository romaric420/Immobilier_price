# PriceWise - Estimation Immobiliere Intelligente

Application de Machine Learning pour estimer les prix immobiliers sur le marche francais.

## Fonctionnalites

- **Analyse de marche** : Vue d'ensemble des prix par ville et type de bien
- **Exploration des donnees** : Facteurs influencant les prix (DPE, etat, localisation)
- **Comparaison de modeles** : 5 algorithmes de regression evalues
- **Estimation personnalisee** : Prediction de prix pour un bien specifique

## Modeles Implementes

| Modele | Type | Caracteristique |
|--------|------|-----------------|
| Regression Lineaire | Regression | Baseline interpretable |
| Ridge Regression | Regression regularisee | Reduction du surapprentissage |
| Lasso Regression | Regression regularisee | Selection de features |
| KNN Regressor (k=5) | Non-parametrique | Approche locale |
| KNN Regressor (k=10) | Non-parametrique | Version stabilisee |

## Metriques d'Evaluation

- R2 Score (coefficient de determination)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Validation croisee (5-fold)
- Analyse des residus

## Donnees

Donnees synthetiques basees sur le marche immobilier francais:
- 10 grandes villes (Paris, Lyon, Marseille, etc.)
- Variables: surface, pieces, DPE, etat, proximite transports
- Prix au m2 realistes par zone

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure du Code

```python
# Fonctions principales
generate_real_estate_data()  # Generation des donnees
train_models()               # Entrainement multi-modeles
prepare_features()           # Encodage et preparation
plot_predictions_vs_actual() # Visualisation performances
```

## Technologies

- Python 3.9+
- Streamlit 1.31
- Scikit-learn 1.4
- Plotly 5.18
- Pandas 2.1
