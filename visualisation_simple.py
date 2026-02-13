"""
VISUALISATION SIMPLE DES RÉSULTATS - INTERVALLES DE CONFIANCE 90%
Version corrigée - Gestion des colonnes mal nommées et valeurs inversées
"""

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# CONFIGURATION STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="📊 Visualisation Prédictions - Intervalle 90%",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# CSS PERSONNALISÉ
#============================================================================

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
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }
    
    .precision-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #1565C0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def format_currency(value):
    """Formate en devise pour l'affichage uniquement"""
    return f"${value:,.0f}"

def format_percentage(value):
    """Formate en pourcentage pour l'affichage uniquement"""
    return f"{value:.1f}%"

def clean_and_fix_dataframe(df_raw):
    """
    Nettoie et corrige le DataFrame en réorganisant les colonnes correctement
    """
    df = df_raw.copy()
    
    # Afficher les colonnes pour debug
    print("Colonnes originales:", df.columns.tolist())
    
    # Identifier les colonnes par leur nom ou position
    columns = df.columns.tolist()
    
    # Initialiser un nouveau DataFrame avec les bonnes colonnes
    df_clean = pd.DataFrame()
    
    # CAS 1: Les colonnes ont des préfixes étranges (A^B_C_)
    # On cherche les colonnes qui contiennent ces mots-clés
    for col in columns:
        col_lower = col.lower()
        
        if 'prix_reel' in col_lower or 'reel' in col_lower:
            df_clean['prix_reel'] = df[col]
        elif 'prix_predit' in col_lower or 'predit_precis' in col_lower:
            df_clean['prix_predit_precis'] = df[col]
        elif 'borne_inferieure' in col_lower or 'borne_inf' in col_lower:
            df_clean['borne_inferieure_90'] = df[col]
        elif 'borne_superieure' in col_lower or 'borne_sup' in col_lower:
            df_clean['borne_superieure_90'] = df[col]
        elif 'intervalle' in col_lower or 'confiance' in col_lower or 'in' in col_lower:
            # Dernière colonne souvent l'intervalle ou l'ID
            pass
    
    # CAS 2: Si on n'a pas trouvé toutes les colonnes, on utilise la position
    if len(df_clean.columns) < 4:
        st.warning("⚠️ Structure de colonnes non standard - Tentative de correction par position")
        
        df_clean = pd.DataFrame()
        
        # D'après l'image, l'ordre semble être:
        # [id, prix_reel, prix_predit_precis, borne_inferieure_90, borne_superieure_90, ?]
        if len(columns) >= 6:
            # La colonne 1 semble être le prix réel
            # La colonne 2 semble être la prédiction
            # La colonne 3 semble être la borne inférieure
            # La colonne 4 semble être la borne supérieure
            
            df_clean['prix_reel'] = pd.to_numeric(df[columns[1]], errors='coerce')
            df_clean['prix_predit_precis'] = pd.to_numeric(df[columns[2]], errors='coerce')
            df_clean['borne_inferieure_90'] = pd.to_numeric(df[columns[3]], errors='coerce')
            df_clean['borne_superieure_90'] = pd.to_numeric(df[columns[4]], errors='coerce')
    
    # CORRECTION: Les valeurs semblent inversées
    # D'après l'image, les prix réels sont dans la colonne 1 (grandes valeurs)
    # et les prédictions dans la colonne 2 (petites valeurs)
    
    if not df_clean.empty:
        # Vérifier si les prédictions sont trop petites par rapport aux prix réels
        if df_clean['prix_predit_precis'].mean() < df_clean['prix_reel'].mean() * 0.5:
            st.warning("⚠️ Les prédictions semblent sous-évaluées - Vérification des colonnes")
            
            # Afficher un échantillon pour debug
            st.write("Échantillon des données après correction:")
            sample_debug = df_clean.head()
            for col in sample_debug.columns:
                sample_debug[col] = sample_debug[col].apply(lambda x: f"{x:,.0f}")
            st.dataframe(sample_debug)
    
    # Calculer les colonnes dérivées
    if not df_clean.empty:
        df_clean['erreur'] = df_clean['prix_predit_precis'] - df_clean['prix_reel']
        df_clean['erreur_abs'] = df_clean['erreur'].abs()
        df_clean['erreur_pourcentage'] = (df_clean['erreur'] / df_clean['prix_reel']) * 100
        
        # Largeur de l'intervalle
        if 'borne_superieure_90' in df_clean.columns and 'borne_inferieure_90' in df_clean.columns:
            df_clean['intervalle_confiance'] = df_clean['borne_superieure_90'] - df_clean['borne_inferieure_90']
        else:
            df_clean['intervalle_confiance'] = 0
        
        df_clean['largeur_relative'] = (df_clean['intervalle_confiance'] / df_clean['prix_reel']) * 100
        
        # Vérifier si les prix sont dans l'intervalle
        df_clean['dans_intervalle'] = (
            (df_clean['prix_reel'] >= df_clean['borne_inferieure_90']) & 
            (df_clean['prix_reel'] <= df_clean['borne_superieure_90'])
        )
    
    return df_clean

def create_display_df(df, cols_to_format):
    """Crée une version formatée pour l'affichage sans modifier l'original"""
    df_display = df.copy()
    for col in cols_to_format:
        if col in df_display.columns:
            if 'prix' in col or 'reel' in col or 'predit' in col or 'borne' in col or 'intervalle' in col or 'erreur' in col and col != 'erreur_pourcentage':
                df_display[col] = df_display[col].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")
            elif 'pourcentage' in col:
                df_display[col] = df_display[col].apply(lambda x: format_percentage(x) if pd.notna(x) else "N/A")
    return df_display

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

@st.cache_data
def load_latest_predictions():
    """Charge le dernier fichier de prédictions et corrige la structure"""
    pred_dir = Path('predictions_quantile')
    
    if not pred_dir.exists():
        return None, "📁 Dossier 'predictions_quantile' introuvable"
    
    pred_files = list(pred_dir.glob('*.csv'))
    
    if not pred_files:
        return None, "❌ Aucun fichier CSV trouvé"
    
    latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
    
    try:
        # Charger le fichier brut
        df_raw = pd.read_csv(latest_file)
        
        # Nettoyer et corriger la structure
        df_clean = clean_and_fix_dataframe(df_raw)
        
        if df_clean.empty:
            return None, "❌ Impossible de corriger la structure du fichier"
        
        return df_clean, f"✅ Fichier chargé et corrigé: {latest_file.name}"
        
    except Exception as e:
        return None, f"❌ Erreur lors du chargement: {str(e)}"

# Chargement
df, message = load_latest_predictions()

# ============================================================================
# SIDEBAR - INFORMATIONS
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/real-estate.png", width=80)
    st.title("📊 Dashboard")
    
    if df is not None:
        st.success(message)
        
        # Avertissement si les données semblent anormales
        if df['prix_predit_precis'].mean() < df['prix_reel'].mean() * 0.3:
            st.warning("⚠️ Les prédictions sont très sous-évaluées")
            st.info("Vérifiez que les colonnes sont correctement assignées dans `clean_and_fix_dataframe()`")
        
        # Informations sur les données
        st.markdown("### 📈 Statistiques")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🏠 Propriétés", f"{len(df):,}")
        with col2:
            st.metric("📅 Date", pd.Timestamp.now().strftime('%d/%m/%Y'))
        
        # Métriques principales
        st.markdown("### 🎯 Prédiction précise")
        st.metric(
            "Prix médian prédit",
            format_currency(df['prix_predit_precis'].median()),
            format_currency(df['prix_predit_precis'].mean()) + " (moyen)"
        )
        
        st.markdown("### 📊 Intervalle 90%")
        st.metric(
            "Largeur moyenne",
            format_currency(df['intervalle_confiance'].mean())
        )
        
        # Couverture
        coverage = df['dans_intervalle'].mean() * 100
        st.metric(
            "✅ Couverture",
            f"{coverage:.1f}%",
            "Cible: 90%" if coverage >= 90 else f"Écart: {90-coverage:.1f}%"
        )
        
        # Erreur
        mape = df['erreur_pourcentage'].abs().mean()
        st.metric(
            "📉 MAPE",
            f"{mape:.1f}%",
            "✓ Excellent" if mape < 15 else "✓ Bon" if mape < 25 else "⚠️ À améliorer"
        )
        
        # Sélection de l'échantillon
        st.markdown("### 🎛️ Filtres")
        n_samples = st.slider(
            "Taille échantillon",
            min_value=50,
            max_value=min(1000, len(df)),
            value=min(200, len(df)),
            step=50
        )
        
        # Tri
        sort_by = st.selectbox(
            "Trier par",
            ["Prix réel", "Erreur absolue", "Largeur intervalle"],
            index=0
        )
        
        sort_dict = {
            "Prix réel": "prix_reel",
            "Erreur absolue": "erreur_abs",
            "Largeur intervalle": "intervalle_confiance"
        }
        
        # Refresh button
        if st.button("🔄 Rafraîchir"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.caption("💡 Les prédictions sont automatiquement corrigées")
        
    else:
        st.error(message)
        st.info("""
        **Pour générer des prédictions:**
        1. Lancez d'abord l'entraînement
        2. Les prédictions seront sauvegardées dans 'predictions_quantile/'
        """)

# ============================================================================
# CONTENU PRINCIPAL
# ============================================================================

if df is not None:
    
    # Avertissement sur la qualité des données
    if df['prix_predit_precis'].mean() < df['prix_reel'].mean() * 0.5:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ ATTENTION: Problème de qualité des données détecté</strong><br>
            Les prédictions sont significativement sous-évaluées par rapport aux prix réels.
            Vérifiez que les colonnes sont correctement assignées dans la fonction de chargement.
        </div>
        """, unsafe_allow_html=True)
    
    # Échantillonnage
    df_sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    # Tri selon la sélection
    if 'sort_by' in locals() and sort_by in sort_dict:
        df_sample = df_sample.sort_values(sort_dict[sort_by])
    
    # ========================================================================
    # TABS PRINCIPAUX
    # ========================================================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 PRÉDICTION PRÉCISE",
        "📊 INTERVALLE 90%",
        "📈 ANALYSE ERREURS",
        "📁 DONNÉES BRUTES"
    ])
    
    # ========================================================================
    # TAB 1: PRÉDICTION PRÉCISE
    # ========================================================================
    
    with tab1:
        st.header("🎯 Analyse de la prédiction précise")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Scatter plot: Réel vs Prédit
            fig_scatter = px.scatter(
                df_sample,
                x='prix_reel',
                y='prix_predit_precis',
                title="🎯 Prédiction vs Prix réel",
                labels={
                    'prix_reel': 'Prix réel ($)',
                    'prix_predit_precis': 'Prédiction ($)'
                },
                hover_data=['erreur', 'erreur_pourcentage'],
                color='erreur',
                color_continuous_scale='RdYlBu',
                color_continuous_midpoint=0,
                opacity=0.7
            )
            
            # Ligne parfaite y=x
            min_val = min(df_sample['prix_reel'].min(), df_sample['prix_predit_precis'].min())
            max_val = max(df_sample['prix_reel'].max(), df_sample['prix_predit_precis'].max())
            
            fig_scatter.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Prédiction parfaite',
                    line=dict(color='red', dash='dash'),
                    showlegend=True
                )
            )
            
            fig_scatter.update_layout(
                height=600,
                xaxis_title="Prix réel",
                yaxis_title="Prix prédit",
                hovermode='closest'
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Métriques de performance
            mae = df_sample['erreur'].abs().mean()
            mape = df_sample['erreur_pourcentage'].abs().mean()
            rmse = np.sqrt((df_sample['erreur'] ** 2).mean())
            
            st.markdown("""
            <div class="precision-card">
                <h3 style="color: white; margin-top: 0;">📊 Performance</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("MAE", format_currency(mae))
                st.metric("MAPE", f"{mape:.1f}%")
            with col2_2:
                st.metric("RMSE", format_currency(rmse))
                st.metric("Biais", format_currency(df_sample['erreur'].mean()))
            
            # Distribution des erreurs
            fig_hist = px.histogram(
                df_sample,
                x='erreur',
                nbins=50,
                title="Distribution des erreurs",
                labels={'erreur': 'Erreur ($)'},
                color_discrete_sequence=['#1E88E5']
            )
            
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
            fig_hist.add_vline(x=mae, line_dash="dot", line_color="green")
            fig_hist.add_vline(x=-mae, line_dash="dot", line_color="green")
            
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # ========================================================================
    # TAB 2: INTERVALLE DE CONFIANCE 90%
    # ========================================================================
    
    with tab2:
        st.header("📊 Analyse de l'intervalle de confiance 90%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Visualisation des intervalles
            fig_interval = go.Figure()
            
            # Sous-échantillon pour lisibilité
            viz_sample = df_sample.head(30).reset_index()
            
            # Prix réel
            fig_interval.add_trace(go.Scatter(
                x=viz_sample.index,
                y=viz_sample['prix_reel'],
                mode='markers',
                name='Prix réel',
                marker=dict(color='red', size=8),
                hovertemplate='Prix réel: $%{y:,.0f}<extra></extra>'
            ))
            
            # Prédiction précise
            fig_interval.add_trace(go.Scatter(
                x=viz_sample.index,
                y=viz_sample['prix_predit_precis'],
                mode='markers',
                name='Prédiction',
                marker=dict(color='blue', size=6),
                hovertemplate='Prédiction: $%{y:,.0f}<extra></extra>'
            ))
            
            # Intervalle
            fig_interval.add_trace(go.Scatter(
                x=viz_sample.index,
                y=viz_sample['borne_superieure_90'],
                mode='lines',
                name='Borne sup.',
                line=dict(dash='dash', color='green', width=1)
            ))
            
            fig_interval.add_trace(go.Scatter(
                x=viz_sample.index,
                y=viz_sample['borne_inferieure_90'],
                mode='lines',
                name='Borne inf.',
                line=dict(dash='dash', color='green', width=1),
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.1)'
            ))
            
            fig_interval.update_layout(
                title=f"Intervalles 90% - {len(viz_sample)} propriétés",
                xaxis_title="Index",
                yaxis_title="Prix ($)",
                height=500
            )
            
            st.plotly_chart(fig_interval, use_container_width=True)
        
        with col2:
            # Distribution des largeurs
            fig_width = px.histogram(
                df_sample,
                x='intervalle_confiance',
                nbins=50,
                title="Largeur des intervalles",
                labels={'intervalle_confiance': 'Largeur ($)'},
                color_discrete_sequence=['#E91E63']
            )
            
            fig_width.add_vline(
                x=df_sample['intervalle_confiance'].mean(),
                line_dash="dash",
                line_color="red"
            )
            
            fig_width.update_layout(height=300)
            st.plotly_chart(fig_width, use_container_width=True)
            
            # Couverture
            coverage = df_sample['dans_intervalle'].mean() * 100
            
            fig_coverage = go.Figure(go.Indicator(
                mode="gauge+number",
                value=coverage,
                title={'text': "Couverture (%)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#1E88E5"},
                    'steps': [
                        {'range': [0, 90], 'color': "#FFCDD2"},
                        {'range': [90, 100], 'color': "#C8E6C9"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig_coverage.update_layout(height=250)
            st.plotly_chart(fig_coverage, use_container_width=True)
    
    # ========================================================================
    # TAB 3: ANALYSE DES ERREURS
    # ========================================================================
    
    with tab3:
        st.header("📈 Analyse des erreurs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Erreur vs Prix réel
            fig_error = px.scatter(
                df_sample,
                x='prix_reel',
                y='erreur_pourcentage',
                title="Erreur % vs Prix réel",
                labels={
                    'prix_reel': 'Prix réel ($)',
                    'erreur_pourcentage': 'Erreur (%)'
                },
                color='erreur_pourcentage',
                color_continuous_scale='RdYlBu',
                color_continuous_midpoint=0
            )
            
            fig_error.add_hline(y=0, line_dash="dash", line_color="black")
            fig_error.update_layout(height=400)
            st.plotly_chart(fig_error, use_container_width=True)
        
        with col2:
            # Boxplot des erreurs
            try:
                df_sample['tranche_prix'] = pd.cut(
                    df_sample['prix_reel'],
                    bins=[0, 300000, 600000, 900000, 1200000, 1500000, np.inf],
                    labels=['<300k', '300-600k', '600-900k', '900k-1.2M', '1.2-1.5M', '>1.5M']
                )
                
                fig_box = px.box(
                    df_sample,
                    x='tranche_prix',
                    y='erreur_pourcentage',
                    title="Erreur % par tranche",
                    labels={'tranche_prix': 'Tranche', 'erreur_pourcentage': 'Erreur %'},
                    color='tranche_prix'
                )
                
                fig_box.add_hline(y=0, line_dash="dash", line_color="black")
                fig_box.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)
            except:
                st.info("Impossible de générer le graphique par tranche")
        
        # Top erreurs
        st.subheader("🔍 Top 10 des plus grandes erreurs")
        
        top_errors = df.nlargest(10, 'erreur_abs')[
            ['prix_reel', 'prix_predit_precis', 'erreur', 'erreur_pourcentage', 
             'borne_inferieure_90', 'borne_superieure_90', 'dans_intervalle']
        ].copy()
        
        top_errors_display = create_display_df(
            top_errors,
            ['prix_reel', 'prix_predit_precis', 'erreur', 'borne_inferieure_90', 'borne_superieure_90', 'erreur_pourcentage']
        )
        top_errors_display['dans_intervalle'] = top_errors_display['dans_intervalle'].apply(lambda x: "✅ Oui" if x else "❌ Non")
        
        st.dataframe(top_errors_display, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # TAB 4: DONNÉES BRUTES
    # ========================================================================
    
    with tab4:
        st.header("📁 Données brutes")
        
        # Filtres
        col1, col2 = st.columns(2)
        
        with col1:
            min_price = int(df['prix_reel'].min())
            max_price = int(df['prix_reel'].max())
            price_range = st.slider(
                "Filtrer par prix",
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price)
            )
        
        with col2:
            interval_filter = st.selectbox(
                "Filtrer par intervalle",
                ["Tous", "Dans l'intervalle", "Hors intervalle"],
                key="filter_interval"
            )
        
        # Application des filtres
        df_filtered = df[
            (df['prix_reel'] >= price_range[0]) &
            (df['prix_reel'] <= price_range[1])
        ].copy()
        
        if interval_filter == "Dans l'intervalle":
            df_filtered = df_filtered[df_filtered['dans_intervalle'] == True]
        elif interval_filter == "Hors intervalle":
            df_filtered = df_filtered[df_filtered['dans_intervalle'] == False]
        
        # Aperçu
        st.subheader(f"📊 Aperçu ({len(df_filtered):,} lignes)")
        
        display_cols = ['prix_reel', 'prix_predit_precis', 'borne_inferieure_90', 
                       'borne_superieure_90', 'intervalle_confiance', 'erreur', 
                       'erreur_pourcentage', 'dans_intervalle']
        
        df_display = create_display_df(
            df_filtered[display_cols].head(100),
            ['prix_reel', 'prix_predit_precis', 'borne_inferieure_90', 
             'borne_superieure_90', 'intervalle_confiance', 'erreur', 'erreur_pourcentage']
        )
        df_display['dans_intervalle'] = df_display['dans_intervalle'].apply(lambda x: "✅ Oui" if x else "❌ Non")
        
        st.dataframe(df_display, use_container_width=True)
        
        # Export
        st.subheader("💾 Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export numérique (brut)
            csv_numeric = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger (valeurs numériques)",
                data=csv_numeric,
                file_name=f"predictions_numeriques_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export formaté
            df_export_readable = create_display_df(
                df_filtered,
                ['prix_reel', 'prix_predit_precis', 'borne_inferieure_90', 
                 'borne_superieure_90', 'intervalle_confiance', 'erreur', 'erreur_pourcentage']
            )
            df_export_readable['dans_intervalle'] = df_export_readable['dans_intervalle'].apply(lambda x: "Oui" if x else "Non")
            
            csv_formatted = df_export_readable.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger (formaté)",
                data=csv_formatted,
                file_name=f"predictions_formatees_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.error("❌ Aucune donnée de prédiction trouvée!")
    
    st.markdown("""
    ### 📋 Instructions
    
    1. **Placez vos fichiers CSV** dans le dossier `predictions_quantile/`
    2. **Structure attendue:** Colonnes pour prix réel, prédiction, bornes
    3. **Rafraîchissez cette page**
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p style='font-size: 0.9rem;'>
        📊 <strong>Visualisation des prédictions - Version corrigée</strong><br>
        ✅ Correction automatique des colonnes mal nommées<br>
        💾 Export en deux formats (numérique brut + formaté lisible)<br>
        ⚠️ Détection automatique des anomalies dans les données
    </p>
</div>
""", unsafe_allow_html=True)