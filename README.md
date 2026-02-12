# 🏠 House Price Prediction Interval

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-5C2D91?style=for-the-badge&logo=lightgbm&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white"/>
  <img src="https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black"/>
</div>

<div align="center">
  <h3>
    <a href="#-aperçu-du-projet">Aperçu</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-modèles-implémentés">Modèles</a> •
    <a href="#-installation">Installation</a> •
    <a href="#-utilisation">Utilisation</a> •
    <a href="#-résultats">Résultats</a>
  </h3>
</div>

---

## 🎯 Aperçu du Projet

**House Price Prediction Interval** est un projet complet de **Machine Learning** dédié à la prédiction des prix immobiliers avec **intervalles de confiance à 90%**. Développé par **Bourzgui Fatima Zahra**, ce projet couvre **41 villes du comté de King, Washington**, et intègre une pipeline complète de la donnée à la visualisation.

### ✨ **Fonctionnalités Clés**

| 🏆 | Fonctionnalité | Description |
|----|---------------|-------------|
| ✅ | **Prédiction avec incertitude** | Intervalles de confiance à 90% via régression quantile |
| ✅ | **6 modèles ML** | LightGBM Quantile, XGBoost, Random Forest, Gradient Boosting, Weighted Ensemble, Model Mix |
| ✅ | **Feature engineering avancé** | 15+ features dérivées automatiquement |
| ✅ | **Dashboard interactif** | Visualisation temps réel avec Streamlit |
| ✅ | **Export Power BI** | Données structurées pour reporting professionnel |
| ✅ | **CI/CD intégré** | GitHub Actions pour tests et déploiement |
| ✅ | **Couverture géographique** | 41 villes du King County, WA |

### 📊 **Performance du Modèle Principal**

| Modèle | MAE | RMSE | R² | Coverage 90% |
|--------|-----|------|----|--------------|
| **Model Mix (Stacking)** | **$41,800** | **$57,500** | **0.901** | **90.1%** |

---

## 👩‍💻 **À Propos de l'Auteur**

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://img.shields.io/badge/Bourzgui%20Fatima%20Zahra-Data%20Scientist-FF6F61?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBkPSJNMTIgMkM2LjQ4IDIgMiA2LjQ4IDIgMTJzNC40OCAxMCAxMCAxMCAxMC00LjQ4IDEwLTEwUzE3LjUyIDIgMTIgMnptMCAzYzEuNjYgMCAzIDEuMzQgMyAzcy0xLjM0IDMtMyAzLTMtMS4zNC0zLTMgMS4zNC0zIDMtM3ptMCAxNGMtMiAzLTMgNC0zIDQtMy0xLTQtMi00LTMgMC0yIDIuMjQtNCA1LTRzNSAyIDUgNGMwIDEtMSAyLTQgM3oiIGZpbGw9IndoaXRlIi8+PC9zdmc+" alt="Author"/>
      </td>
    </tr>
    <tr>
      <td align="center">
        <strong>Bourzgui Fatima Zahra</strong><br/>
        Data Analyst 
      </td>
    </tr>
  </table>
</div>

**Contact :**
- 📧 Email : [bourzguifatimazahra@gmail.com](mailto:bourzguifatimazahra@gmail.com)
- 🔗 LinkedIn : [Bourzgui Fatima Zahra](https://www.linkedin.com/in/fatimazahrabourzgui/)
- 💻 GitHub : [@bourzguifatimazahra](https://github.com/Bourzguifatimazahra)
- 📍 Localisation : Casablanca, Maroc

---
 
## 🤖 Modèles Implémentés

### 📊 **Comparaison des Performances**

<div align="center">
  <table>
    <thead>
      <tr>
        <th>Modèle</th>
        <th>Type</th>
        <th>MAE</th>
        <th>RMSE</th>
        <th>R²</th>
        <th>Coverage 90%</th>
        <th>⚡ Statut</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>LightGBM Quantile</strong></td>
        <td>Quantile Regression</td>
        <td>$42,350</td>
        <td>$58,200</td>
        <td>0.892</td>
        <td>89.2%</td>
        <td>✅ Principal</td>
      </tr>
      <tr>
        <td><strong>Model Mix (Stacking)</strong></td>
        <td>Ensemble</td>
        <td><strong>$41,800</strong></td>
        <td><strong>$57,500</strong></td>
        <td><strong>0.901</strong></td>
        <td><strong>90.1%</strong></td>
        <td>🏆 Champion</td>
      </tr>
      <tr>
        <td>XGBoost</td>
        <td>Gradient Boosting</td>
        <td>$45,800</td>
        <td>$62,100</td>
        <td>0.874</td>
        <td>86.5%</td>
        <td>✅ Actif</td>
      </tr>
      <tr>
        <td>Random Forest</td>
        <td>Bagging</td>
        <td>$48,200</td>
        <td>$65,800</td>
        <td>0.858</td>
        <td>85.1%</td>
        <td>✅ Actif</td>
      </tr>
      <tr>
        <td>Gradient Boosting</td>
        <td>Gradient Boosting</td>
        <td>$46,900</td>
        <td>$63,500</td>
        <td>0.869</td>
        <td>86.8%</td>
        <td>✅ Actif</td>
      </tr>
      <tr>
        <td>Weighted Ensemble</td>
        <td>Moyenne pondérée</td>
        <td>$44,100</td>
        <td>$60,200</td>
        <td>0.883</td>
        <td>87.9%</td>
        <td>✅ Actif</td>
      </tr>
    </tbody>
  </table>
</div>

### 🎯 **LightGBM Quantile - Modèle Principal**

```python
# Configuration du modèle quantile
quantile_models = {
    0.05: "Borne inférieure (intervalle 90%)",
    0.50: "Prédiction médiane", 
    0.95: "Borne supérieure (intervalle 90%)"
}

# Entraînement
params = {
    'objective': 'quantile',
    'metric': 'quantile',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'early_stopping_rounds': 50
}
```

### 🔬 **Model Mix (Stacking) - Champion**

**Composition optimale :**
- 🟢 **40% XGBoost** - Robustesse aux outliers
- 🔵 **30% Random Forest** - Gestion non-linéarités
- 🟠 **30% Gradient Boosting** - Précision fine

---

## 📊 Métriques d'Évaluation

### 📈 **Métriques d'Intervalles de Prédiction**

| Métrique | Formule | Objectif | Notre Score |
|----------|---------|---------|-------------|
| **Coverage Rate** | `1/n * ∑ 𝟙(y_i ∈ [L_i, U_i])` | ≥ 90% | **90.1%** |
| **Interval Width** | `1/n * ∑ (U_i - L_i)` | Minimiser | **$178,000** |
| **Pinball Loss** | `∑ (τ - 𝟙(y < q)) * (y - q)` | Minimiser | **0.043** |
| **Interval Score** | Score de Winkler | Minimiser | **Optimal** |

### 📉 **Métriques de Régression**

| Métrique | Description | Score |
|----------|-------------|-------|
| **MAE** | Mean Absolute Error | **$41,800** |
| **RMSE** | Root Mean Square Error | **$57,500** |
| **MAPE** | Mean Absolute Percentage Error | **12.3%** |
| **R²** | Coefficient de détermination | **0.901** |

---

## 📁 Structure des Données

### **Dataset Original** - 21,460 propriétés

| Colonne | Type | Description |
|---------|------|-------------|
| `sale_price` | float | Prix de vente (cible) |
| `sale_date` | datetime | Date de vente |
| `city` | string | Ville (41 valeurs uniques) |
| `sqft` | float | Surface habitable |
| `sqft_lot` | float | Surface terrain |
| `beds` | int | Nombre de chambres |
| `bath_full` | int | Salles de bain complètes |
| `grade` | int | Note construction (1-13) |
| `condition` | int | État (1-5) |
| `year_built` | int | Année construction |

### 🔧 **Features Dérivées (15+)**

```python
# Features créées automatiquement
features_derivees = {
    'property_age': '2024 - year_built',
    'since_reno': '2024 - year_reno',
    'imp_land_ratio': 'imp_val / land_val',
    'sqft_ratio': 'sqft / sqft_lot',
    'log_sqft': 'log1p(sqft)',
    'log_price': 'log1p(sale_price)',
    'total_bathrooms': 'bath_full + bath_3qtr*0.75 + bath_half*0.5',
    'has_garage': 'garage_sqft > 0',
    'has_view': 'total_views > 0',
    'price_per_sqft': 'sale_price / sqft'
}
```

### 🏙️ **Couverture Géographique**

<details>
<summary><b>📌 41 villes du King County, WA (Cliquez pour déplier)</b></summary>
<br>

| Région | Villes |
|--------|--------|
| **Seattle Metro** | Seattle, Bellevue, Redmond, Kirkland, Renton |
| **Eastside** | Sammamish, Issaquah, Mercer Island, Medina, Clyde Hill, Yarrow Point |
| **South King** | Kent, Auburn, Federal Way, Des Moines, SeaTac, Tukwila |
| **North King** | Shoreline, Kenmore, Bothell, Woodinville, Lake Forest Park |
| **Snoqualmie Valley** | Snoqualmie, North Bend, Carnation, Duvall |
| **Vashon Island** | Vashon, Maury Island |
| **Rural East** | Enumclaw, Black Diamond, Maple Valley, Covington |
| **Other** | Algona, Beaux Arts, Burien, Hunts Point, Normandy Park, Pacific, Skykomish |

**Total: 41 villes uniques**
</details>

---

## ⚙️ Installation

### 📋 **Prérequis**
- Python 3.9 ou supérieur
- Git
- 8GB RAM minimum recommandé

### 🚀 **Installation Rapide**

```bash
# 1. Cloner le dépôt
git clone https://github.com/Bourzguifatimazahra/house-price-pi.git
cd house-price-pi

# 2. Créer l'environnement virtuel
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Vérifier l'installation
python pipeline_complete.py --help
```

### 📦 **Dépendances Principales**

```txt
# Core ML
lightgbm==4.0.0
xgboost==2.0.0
scikit-learn==1.3.0

# Data Processing
pandas==2.0.0
numpy==1.24.0
scipy==1.11.0

# Visualization
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.17.0
streamlit==1.28.0

# Utils
joblib==1.3.0
pyyaml==6.0
tqdm==4.65.0
```

---

## 🚀 Utilisation

### 🎮 **Makefile - Commandes Principales**

```bash
# Afficher l'aide
make help

# Pipeline complet (recommandé)
make pipeline

# Commandes individuelles
make data        # Générer les données synthétiques
make features    # Feature engineering
make train       # Entraîner tous les modèles
make train-quantile # Entraîner LightGBM Quantile
make predict     # Générer les prédictions
make dashboard   # Lancer le dashboard Streamlit
make test        # Exécuter les tests
make lint        # Vérifier la qualité du code
make clean       # Nettoyer les artifacts
```

### 🐍 **Pipeline Python**

```python
# pipeline_complete.py - Exécution complète
from src.models.quantile_trainer import QuantileTrainer
from src.models.predict import PredictionPipeline
from src.evaluation.metrics import ModelEvaluator

# 1. Entraînement des modèles quantiles
trainer = QuantileTrainer()
metrics = trainer.run_training_pipeline('data/raw/dataset.csv')

# 2. Prédictions avec intervalles
pipeline = PredictionPipeline()
predictions = pipeline.run_prediction_pipeline('data/raw/dataset.csv')

# 3. Évaluation
coverage = ModelEvaluator.calculate_coverage_rate(
    predictions['sale_price'],
    predictions['lower_bound'],
    predictions['upper_bound']
)
print(f"✅ Coverage à 90%: {coverage:.1%}")
```

## 📈 Résultats Détaillés

### 🏆 **Performance par Segment de Prix**

| Segment | Prix Moyen | MAE | Coverage | Width | Width % |
|---------|------------|-----|----------|-------|---------|
| **Budget** (< $500k) | $425,000 | $28,900 | 91.2% | $152,000 | 35.8% |
| **Mid-Range** ($500k-1M) | $785,000 | $41,200 | 90.5% | $188,000 | 23.9% |
| **Premium** ($1M-2M) | $1,450,000 | $68,500 | 88.7% | $245,000 | 16.9% |
| **Luxury** (> $2M) | $2,850,000 | $112,000 | 85.3% | $320,000 | 11.2% |

### 📊 **Analyse par Ville - Top 5**

| Ville | Propriétés | Prix Moyen | MAE | R² | Coverage |
|-------|------------|-----------|-----|----|----------|
| **Medina** | 342 | $2,450,000 | $98,000 | 0.87 | 88.5% |
| **Bellevue** | 1,245 | $1,280,000 | $52,000 | 0.89 | 89.8% |
| **Seattle** | 4,210 | $875,000 | $38,000 | 0.91 | 90.5% |
| **Redmond** | 987 | $945,000 | $41,000 | 0.90 | 90.2% |
| **Renton** | 856 | $615,000 | $29,000 | 0.88 | 89.7% |

### 📉 **Feature Importance (LightGBM)**

```
1.  log_sqft        ████████████ 100.0%
2.  property_age    ████████      78.3%
3.  grade           ███████       72.1%
4.  sqft_ratio      ██████        63.5%
5.  total_bathrooms █████         54.8%
6.  has_view        ████          42.2%
7.  condition       ███           35.7%
8.  since_reno      ██            21.3%
9.  sqft_lot        ██            18.9%
10. imp_land_ratio  █             12.4%
```

---

## 🖼️ Visualisations

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Distribution des Prix</b></td>
      <td align="center"><b>Matrice de Corrélation</b></td>
    </tr>
    <tr>
      <td><img src="artifacts/price_distribution.png" width="400"/></td>
      <td><img src="artifacts/correlation_matrix.png" width="400"/></td>
    </tr>
    <tr>
      <td align="center"><b>Intervalles de Prédiction</b></td>
      <td align="center"><b>Analyse par Ville</b></td>
    </tr>
    <tr>
      <td><img src="artifacts/prediction_intervals.png" width="400"/></td>
      <td><img src="artifacts/city_analysis.png" width="400"/></td>
    </tr>
  </table>
</div>

---

## 🔬 Analyse Approfondie

### 📌 **Défis Relevés**

1. **Hétéroscédasticité** - Variance non-constante des prix
   - ✅ Solution: Régression quantile pour capturer l'incertitude

2. **Données déséquilibrées par ville**
   - ✅ Solution: Features géographiques et stratification

3. **Valeurs aberrantes**
   - ✅ Solution: Détection IQR + transformation log

4. **Intervalles trop larges**
   - ✅ Solution: Optimisation des hyperparamètres

### 💡 **Innovations**

- **Feature engineering géographique**: Clustering spatial des propriétés
- **Stacking adaptatif**: Poids dynamiques selon le segment de prix
- **Validation croisée temporelle**: Respect de la chronologie des ventes
- **Export multi-format**: CSV, Excel, Parquet, Power BI

---

## 🧪 Tests et Qualité

```bash
# Exécuter tous les tests
make test

# Vérifier la couverture
pytest tests/ --cov=src --cov-report=html

# Linting
make lint

# Formatage automatique
make format
```

**Couverture de code:** > 85%

---

## 📚 Documentation

### 📖 **Notebooks Jupyter**

| Notebook | Description | Lien |
|----------|-------------|------|
| `01_eda.ipynb` | Analyse exploratoire des données | [Voir](01_eda.ipynb) |
| `02_feature_engineering.ipynb` | Création des features | [Voir](02_feature_engineering.ipynb) |

## 🤝 Contribution

Je suis ouverte aux collaborations et suggestions pour améliorer ce projet !

### 📝 **Comment contribuer**

1. **Fork** le projet
2. **Créer une branche** (`git checkout -b feature/amazing-feature`)
3. **Commit** les changements (`git commit -m 'Add amazing feature'`)
4. **Push** (`git push origin feature/amazing-feature`)
5. **Ouvrir une Pull Request**

```

Copyright (c) 2026 Bourzgui Fatima Zahra

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 🙏 Remerciements

Je tiens à remercier :

- **King County Department of Assessments** - Pour la mise à disposition des données
- **LightGBM, XGBoost, Scikit-learn** - Pour leurs bibliothèques exceptionnelles
- **Streamlit** - Pour le framework de dashboard
- **Communauté Open Source** - Pour le partage de connaissances

---

## 📊 Badges et Statistiques

<div align="center">
  
  ![GitHub stars](https://img.shields.io/github/stars/bourzgui-fatimazahra/house-price-pi?style=social)
  ![GitHub forks](https://img.shields.io/github/forks/bourzgui-fatimazahra/house-price-pi?style=social)
  ![GitHub watchers](https://img.shields.io/github/watchers/bourzgui-fatimazahra/house-price-pi?style=social)
  
  ![GitHub last commit](https://img.shields.io/github/last-commit/bourzgui-fatimazahra/house-price-pi)
  ![GitHub repo size](https://img.shields.io/github/repo-size/bourzgui-fatimazahra/house-price-pi)
  ![GitHub license](https://img.shields.io/github/license/bourzgui-fatimazahra/house-price-pi)
  
  ![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
  ![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)
  ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
  ![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
  
</div>

---

## 📞 Contact

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="mailto:bourzguifatimazahra@gmail.com">
          <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white"/>
        </a></td>
    </tr>
  </table>
</div>

**Bourzgui Fatima Zahra**

---

<div align="center">
  <h3>
    ⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐
  </h3>
  <p>
    Développé avec ❤️ par Bourzgui Fatima Zahra
  </p>
  <p>
    <sub>© 2026 House Price Prediction Interval. Tous droits réservés.</sub>
  </p>
</div>

---

*Dernière mise à jour : 11 Février 2026*
