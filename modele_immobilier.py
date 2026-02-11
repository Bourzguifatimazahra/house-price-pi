"""
PRÉDICTION PRÉCISE AVEC INTERVALLES DE CONFIANCE 90%
- Quantiles: 0.05 (borne inf), 0.5 (médiane), 0.95 (borne sup)
- LightGBM Quantile Regression
- Visualisation des intervalles
- Export CSV avec prédiction + bornes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

# ============================================================================
# DATA PREPROCESSOR - FILTRAGE PRIX 0$
# ============================================================================

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        
        # Créer sale_price
        if 'sale_price' not in df.columns:
            df['sale_price'] = df['land_val'] + df['imp_val']
        
        # 🚨 FILTRAGE - Supprimer les prix à 0$ et négatifs
        initial = len(df)
        df = df[df['sale_price'] > 1000]
        
        print(f"\n{'='*60}")
        print(f"📊 FILTRAGE DES DONNÉES")
        print(f"{'='*60}")
        print(f"   Lignes avant: {initial:,}")
        print(f"   Lignes après: {len(df):,}")
        print(f"   Supprimées: {initial - len(df):,} (prix ≤ 1000$)")
        print(f"{'='*60}")
        print(f"\n✅ Données: {len(df):,} lignes")
        print(f"   Prix moyen: ${df['sale_price'].mean():,.0f}")
        print(f"   Prix min: ${df['sale_price'].min():,.0f}")
        print(f"   Prix max: ${df['sale_price'].max():,.0f}")
        
        return df
    
    def engineer_features(self, df):
        df = df.copy()
        current_year = datetime.now().year
        
        # Total baths
        df['total_baths'] = 0
        if 'bath_full' in df.columns: df['total_baths'] += df['bath_full'].fillna(0)
        if 'bath_half' in df.columns: df['total_baths'] += df['bath_half'].fillna(0) * 0.5
        df['total_baths'] = df['total_baths'].clip(1, 6)
        
        # Age
        if 'year_built' in df.columns:
            df['year_built'] = df['year_built'].fillna(current_year - 30)
            df['age'] = (current_year - df['year_built']).clip(0, 100)
            df['age_sq'] = df['age'] ** 2
        
        # Sqft
        if 'sqft' in df.columns:
            df['sqft'] = df['sqft'].fillna(df['sqft'].median())
            df['sqft_log'] = np.log1p(df['sqft'])
            df['sqft_sq'] = df['sqft'] ** 2
        
        # Beds
        if 'beds' in df.columns:
            df['beds'] = df['beds'].fillna(3).clip(1, 10)
            df['beds_sq'] = df['beds'] ** 2
        
        # Garage
        if 'gara_sqft' in df.columns:
            df['has_garage'] = (df['gara_sqft'].fillna(0) > 0).astype(int)
        
        # Views
        view_cols = [c for c in df.columns if c.startswith('view_')]
        if view_cols:
            df['total_views'] = df[view_cols].sum(axis=1)
            df['has_view'] = (df['total_views'] > 0).astype(int)
        
        # Grade
        if 'grade' in df.columns:
            df['grade'] = df['grade'].fillna(7).clip(1, 13)
        
        # Drop useless
        drop_cols = ['id', 'sale_nbr', 'join_status', 'submarket', 'join_year',
                    'present_use', 'subdivision', 'zoning', 'sale_date', 'sale_warning']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        print(f"✅ Features: {df.shape[1]} colonnes")
        return df
    
    def prepare_features(self, df, is_training=True):
        df = df.copy()
        
        if 'sale_price' in df.columns:
            y = df['sale_price']
            X = df.drop(columns=['sale_price'])
        else:
            y = None
            X = df
        
        # Colonnes catégorielles
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna('MISSING')
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col])
                elif col in self.label_encoders:
                    known = set(self.label_encoders[col].classes_)
                    X[col] = X[col].apply(lambda x: 
                        self.label_encoders[col].transform([x])[0] if x in known else -1)
                else:
                    X = X.drop(columns=[col])
        
        # Conversion numérique
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Standardisation
        if is_training:
            self.feature_columns = X.columns.tolist()
            X_scaled = self.scaler.fit_transform(X)
        else:
            for col in self.feature_columns:
                if col not in X.columns: X[col] = 0
            X = X[self.feature_columns]
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=self.feature_columns), y

# ============================================================================
# LIGHTGBM QUANTILE - PRÉDICTION PRÉCISE AVEC INTERVALLES 90%
# ============================================================================

class QuantilePredictor:
    def __init__(self, seed=42):
        self.seed = seed
        self.models = {}
        
    def train(self, X_train, y_train, X_val, y_val):
        """Entraîne LightGBM pour les quantiles 0.05, 0.5, 0.95"""
        print("\n" + "="*70)
        print("🎯 LIGHTGBM QUANTILE REGRESSION - INTERVALLE 90%")
        print("="*70)
        
        quantiles = [0.05, 0.5, 0.95]
        self.models = {}
        
        for q in quantiles:
            print(f"\n📊 Entraînement quantile {int(q*100)}%...")
            
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=63,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=self.seed,
                verbose=-1
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            self.models[f'q_{int(q*100)}'] = model
            print(f"   ✅ Quantile {int(q*100)}% - Meilleure itération: {model.best_iteration_}")
        
        return self.models
    
    def predict(self, X, return_all=True):
        """Prédiction avec intervalles de confiance 90%"""
        y_lower = self.models['q_5'].predict(X)
        y_median = self.models['q_50'].predict(X)
        y_upper = self.models['q_95'].predict(X)
        
        if return_all:
            return y_lower, y_median, y_upper
        else:
            return y_median  # Prédiction précise (médiane)

# ============================================================================
# PIPELINE PRINCIPAL - PRÉDICTION PRÉCISE
# ============================================================================

class HousePricePipeline:
    def __init__(self, seed=42):
        self.preprocessor = DataPreprocessor()
        self.predictor = QuantilePredictor(seed)
        self.results = {}
        
    def train(self, data_path='dataset.csv'):
        print("\n" + "="*80)
        print("🏠 PRÉDICTION PRÉCISE AVEC INTERVALLE DE CONFIANCE 90%")
        print("="*80)
        print(f"📁 Fichier: {data_path}")
        print("="*80)
        
        # 1. Chargement avec filtrage
        df = self.preprocessor.load_data(data_path)
        
        # 2. Feature engineering
        df = self.preprocessor.engineer_features(df)
        
        # 3. Préparation
        X, y = self.preprocessor.prepare_features(df, is_training=True)
        
        # 4. Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        print(f"\n📊 Split: Train={len(X_train):,} Val={len(X_val):,} Test={len(X_test):,}")
        
        # 5. Entraînement quantile
        self.predictor.train(X_train, y_train, X_val, y_val)
        
        # 6. Évaluation
        self.evaluate(X_test, y_test)
        
        # 7. Export
        self.export_predictions(X_test, y_test)
        self.save_models()
        
        return self.results
    
    def evaluate(self, X_test, y_test):
        """Évaluation des prédictions avec intervalles"""
        target_mean = y_test.mean()
        
        # Prédictions
        y_lower, y_median, y_upper = self.predictor.predict(X_test)
        
        # Métriques
        mae = mean_absolute_error(y_test, y_median)
        rmse = np.sqrt(mean_squared_error(y_test, y_median))
        r2 = r2_score(y_test, y_median)
        mape = (mae / target_mean) * 100
        
        # Métriques d'intervalle
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper)) * 100
        interval_width = np.mean(y_upper - y_lower)
        
        self.results = {
            'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape,
            'coverage': coverage, 'interval_width': interval_width,
            'y_true': y_test.values, 'y_median': y_median,
            'y_lower': y_lower, 'y_upper': y_upper
        }
        
        print("\n" + "="*70)
        print("📊 RÉSULTATS - PRÉDICTION PRÉCISE (MÉDIANE) + INTERVALLE 90%")
        print("="*70)
        print(f"\n🎯 PRÉDICTION PRÉCISE (Quantile 50%):")
        print(f"   MAE:  ${mae:,.0f} ({mape:.1f}%)")
        print(f"   RMSE: ${rmse:,.0f}")
        print(f"   R²:   {r2:.4f}")
        print(f"\n📉 INTERVALLE DE CONFIANCE 90% (Quantiles 5% - 95%):")
        print(f"   Coverage: {coverage:.1f}% des valeurs dans l'intervalle")
        print(f"   Largeur moyenne: ${interval_width:,.0f}")
        print(f"   Borne inférieure moyenne: ${y_lower.mean():,.0f}")
        print(f"   Borne supérieure moyenne: ${y_upper.mean():,.0f}")
        print("="*70)
        
        return self.results
    
    def export_predictions(self, X_test, y_test):
        """Export CSV avec prédiction précise + intervalles"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        Path('predictions_quantile').mkdir(exist_ok=True)
        
        # Prédictions
        y_lower, y_median, y_upper = self.predictor.predict(X_test)
        
        # 🎯 DataFrame avec prédiction précise ET intervalles
        df_pred = pd.DataFrame({
            'id': range(1, len(y_test)+1),
            'prix_reel': y_test.values,
            'prix_predit_precis': y_median,  # ✅ Prédiction précise (médiane)
            'borne_inferieure_90': y_lower,   # 📉 Quantile 5%
            'borne_superieure_90': y_upper,   # 📈 Quantile 95%
            'intervalle_confiance': y_upper - y_lower,
            'erreur': y_test.values - y_median,
            'erreur_pourcentage': np.where(
                y_test.values > 0,
                ((y_test.values - y_median) / y_test.values) * 100,
                0
            ),
            'dans_intervalle': ((y_test.values >= y_lower) & (y_test.values <= y_upper))
        })
        
        # Sauvegarde
        file_path = f'predictions_quantile/prediction_precise_intervalle90_{timestamp}.csv'
        df_pred.to_csv(file_path, index=False)
        
        print(f"\n✅ EXPORT RÉUSSI: {file_path}")
        print(f"   {len(df_pred):,} prédictions précises avec intervalles 90%")
        print(f"   Colonnes:")
        print(f"   - prix_predit_precis : Prédiction précise (médiane)")
        print(f"   - borne_inferieure_90 : Borne inférieure (quantile 5%)")
        print(f"   - borne_superieure_90 : Borne supérieure (quantile 95%)")
        
        return df_pred
    
    def save_models(self):
        """Sauvegarde les modèles quantiles"""
        Path('models_quantile').mkdir(exist_ok=True)
        t = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, model in self.predictor.models.items():
            joblib.dump(model, f'models_quantile/lightgbm_{name}_{t}.joblib')
        
        joblib.dump(self.preprocessor, f'models_quantile/preprocessor_{t}.joblib')
        print(f"\n✅ Modèles quantiles sauvegardés dans models_quantile/")

# ============================================================================
# STREAMLIT - VISUALISATION PRÉDICTION PRÉCISE + INTERVALLES
# ============================================================================

def run_streamlit():
    """Interface Streamlit - Prédiction précise avec intervalles 90%"""
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.set_page_config(page_title="🎯 Prédiction Précise avec Intervalle 90%", layout="wide")
    
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            color: #1E88E5;
            text-align: center;
            padding: 1.5rem;
            background: linear-gradient(90deg, #E3F2FD, #BBDEFB);
            border-radius: 15px;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        .precision-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">🎯 PRÉDICTION PRÉCISE AVEC INTERVALLE DE CONFIANCE 90%</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/real-estate.png", width=100)
        st.title("📊 Quantiles")
        st.info("""
        **Configuration:**
        - 🎯 **Quantile 50%**: Prédiction précise
        - 📉 **Quantile 5%**: Borne inférieure
        - 📈 **Quantile 95%**: Borne supérieure
        - 📊 **Intervalle 90%**: Entre 5% et 95%
        """)
        
        # Charger les prédictions
        pred_files = sorted(Path('predictions_quantile').glob('prediction_precise_intervalle90_*.csv'), reverse=True)
        
        if pred_files:
            df = pd.read_csv(pred_files[0])
            st.success(f"✅ {len(df):,} prédictions chargées")
            
            # Métriques clés
            st.metric("🎯 Prédiction précise (médiane)", 
                     f"${df['prix_predit_precis'].mean():,.0f}")
            st.metric("📉 Borne inférieure moyenne", 
                     f"${df['borne_inferieure_90'].mean():,.0f}")
            st.metric("📈 Borne supérieure moyenne", 
                     f"${df['borne_superieure_90'].mean():,.0f}")
            st.metric("📊 Largeur intervalle", 
                     f"${df['intervalle_confiance'].mean():,.0f}")
            st.metric("✅ Coverage", 
                     f"{df['dans_intervalle'].mean()*100:.1f}%")
        else:
            st.warning("Aucune prédiction trouvée")
            df = None
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 PRÉDICTION PRÉCISE",
        "📊 INTERVALLES 90%",
        "📈 ANALYSE",
        "📁 EXPORT"
    ])
    
    with tab1:
        st.header("🎯 PRÉDICTION PRÉCISE (Quantile 50%)")
        
        if df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot - Prédiction précise vs réel
                fig = px.scatter(
                    df.sample(min(1000, len(df))),
                    x='prix_reel',
                    y='prix_predit_precis',
                    title="Prédiction précise vs Prix réel",
                    labels={'prix_reel': 'Prix réel ($)', 
                           'prix_predit_precis': 'Prédiction précise ($)'},
                    trendline="ols"
                )
                fig.add_shape(
                    type="line", line=dict(dash="dash", color="red"),
                    x0=df['prix_reel'].min(), y0=df['prix_reel'].min(),
                    x1=df['prix_reel'].max(), y1=df['prix_reel'].max()
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Métriques de précision
                mae = df['erreur'].abs().mean()
                mape = df['erreur_pourcentage'].abs().mean()
                
                st.markdown(f"""
                <div class="precision-card">
                    <h3 style="color: white; margin-top: 0;">🎯 Performance de la prédiction précise</h3>
                    <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">MAE: ${mae:,.0f}</p>
                    <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">MAPE: {mape:.1f}%</p>
                    <p style="font-size: 1.2rem; margin-bottom: 0;">R²: {df['prix_reel'].corr(df['prix_predit_precis'])**2:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Distribution des erreurs
                fig = px.histogram(
                    df, x='erreur',
                    nbins=50,
                    title="Distribution des erreurs de prédiction",
                    labels={'erreur': 'Erreur ($)'},
                    color_discrete_sequence=['#1E88E5']
                )
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📊 INTERVALLE DE CONFIANCE 90% (Quantiles 5% - 95%)")
        
        if df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Échantillon avec intervalles
                sample = df.sample(min(100, len(df))).sort_values('prix_reel')
                
                fig = go.Figure()
                
                # Prix réel
                fig.add_trace(go.Scatter(
                    x=sample.index, y=sample['prix_reel'],
                    mode='markers',
                    name='Prix réel',
                    marker=dict(color='red', size=8)
                ))
                
                # Prédiction précise
                fig.add_trace(go.Scatter(
                    x=sample.index, y=sample['prix_predit_precis'],
                    mode='markers+lines',
                    name='Prédiction précise',
                    marker=dict(color='blue', size=6),
                    line=dict(color='blue', width=1)
                ))
                
                # Intervalle 90%
                fig.add_trace(go.Scatter(
                    x=sample.index, y=sample['borne_superieure_90'],
                    mode='lines',
                    name='Borne supérieure (95%)',
                    line=dict(dash='dash', color='green')
                ))
                
                fig.add_trace(go.Scatter(
                    x=sample.index, y=sample['borne_inferieure_90'],
                    mode='lines',
                    name='Borne inférieure (5%)',
                    line=dict(dash='dash', color='green'),
                    fill='tonexty',
                    fillcolor='rgba(0,255,0,0.1)'
                ))
                
                fig.update_layout(
                    title="Intervalle de confiance 90% - Échantillon de 100 propriétés",
                    xaxis_title="Index",
                    yaxis_title="Prix ($)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Couverture par tranche de prix
                df['tranche'] = pd.cut(
                    df['prix_reel'],
                    bins=[0, 300000, 600000, 900000, 1200000, 1500000, 2000000, np.inf],
                    labels=['<300k', '300-600k', '600-900k', '900k-1.2M', '1.2-1.5M', '1.5-2M', '>2M']
                )
                
                coverage_by_price = df.groupby('tranche')['dans_intervalle'].mean() * 100
                
                fig = px.bar(
                    x=coverage_by_price.index,
                    y=coverage_by_price.values,
                    title="Taux de couverture par tranche de prix",
                    labels={'x': 'Tranche de prix', 'y': 'Couverture (%)'},
                    text=[f"{v:.1f}%" for v in coverage_by_price.values],
                    color_discrete_sequence=['#1E88E5']
                )
                fig.add_hline(y=90, line_dash="dash", line_color="red")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("📈 ANALYSE DÉTAILLÉE")
        
        if df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution des largeurs d'intervalle
                fig = px.histogram(
                    df, x='intervalle_confiance',
                    nbins=50,
                    title="Distribution de la largeur des intervalles",
                    labels={'intervalle_confiance': 'Largeur intervalle ($)'},
                    color_discrete_sequence=['#E91E63']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Largeur relative vs prix
                df['largeur_relative'] = (df['intervalle_confiance'] / df['prix_reel']) * 100
                
                fig = px.scatter(
                    df.sample(min(500, len(df))),
                    x='prix_reel',
                    y='largeur_relative',
                    title="Largeur relative de l'intervalle vs Prix",
                    labels={'prix_reel': 'Prix réel ($)', 
                           'largeur_relative': 'Largeur relative (%)'},
                    color_discrete_sequence=['#4CAF50']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques descriptives
            st.subheader("📊 Statistiques des prédictions")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Prix réel moyen", f"${df['prix_reel'].mean():,.0f}")
            with col2:
                st.metric("Prédiction précise", f"${df['prix_predit_precis'].mean():,.0f}")
            with col3:
                st.metric("Erreur absolue moyenne", f"${df['erreur'].abs().mean():,.0f}")
            with col4:
                st.metric("Erreur relative", f"{df['erreur_pourcentage'].abs().mean():.1f}%")
    
    with tab4:
        st.header("📁 EXPORT DES PRÉDICTIONS PRÉCISES")
        
        pred_files = list(Path('predictions_quantile').glob('prediction_precise_intervalle90_*.csv'))
        
        if pred_files:
            for f in sorted(pred_files, reverse=True)[:10]:
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"📄 {f.name}")
                with col2:
                    size = f.stat().st_size / 1024
                    st.write(f"{size:.1f} KB")
                with col3:
                    df_temp = pd.read_csv(f)
                    st.write(f"{len(df_temp):,} lignes")
                with col4:
                    with open(f, "rb") as file:
                        st.download_button("📥", file, f.name, key=f.name)
            
            # Aperçu
            st.subheader("👁️ Aperçu des données (10 premières lignes)")
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.info("Aucune prédiction trouvée. Lancez d'abord l'entraînement.")

# ============================================================================
# MAIN
# ============================================================================

def main():
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'streamlit':
            run_streamlit()
        elif sys.argv[1] == 'train':
            data_file = sys.argv[2] if len(sys.argv) > 2 else 'dataset.csv'
            HousePricePipeline().train(data_file)
    else:
        HousePricePipeline().train('dataset.csv')

if __name__ == "__main__":
    main()
    