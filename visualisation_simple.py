"""
VISUALISATION SIMPLE DES RÉSULTATS - INTERVALLES DE CONFIANCE 90%
Charge directement le dernier fichier CSV de prédictions et affiche les graphiques
Version corrigée - sans erreur 'erreur_abs'
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
# ============================================================================

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
# TITRE PRINCIPAL
# ============================================================================

st.markdown('<div class="main-header">📊 VISUALISATION DES PRÉDICTIONS - INTERVALLE DE CONFIANCE 90%</div>', 
            unsafe_allow_html=True)

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

@st.cache_data
def load_latest_predictions():
    """Charge le dernier fichier de prédictions"""
    pred_dir = Path('predictions_quantile')
    
    if not pred_dir.exists():
        return None, "📁 Dossier 'predictions_quantile' introuvable"
    
    pred_files = list(pred_dir.glob('prediction_precise_intervalle90_*.csv'))
    
    if not pred_files:
        return None, "❌ Aucun fichier de prédictions trouvé"
    
    latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    # Ajouter les colonnes calculées nécessaires
    df['erreur_abs'] = df['erreur'].abs()
    df['largeur_relative'] = (df['intervalle_confiance'] / df['prix_reel']) * 100
    
    return df, f"✅ Fichier chargé: {latest_file.name}"

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
            f"${df['prix_predit_precis'].median():,.0f}",
            f"${df['prix_predit_precis'].mean():,.0f} (moyen)"
        )
        
        st.markdown("### 📊 Intervalle 90%")
        st.metric(
            "Largeur moyenne",
            f"${df['intervalle_confiance'].mean():,.0f}"
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
        st.caption("💡 Les prédictions sont chargées depuis le dossier 'predictions_quantile/'")
        
    else:
        st.error(message)
        st.info("""
        **Pour générer des prédictions:**
        1. Lancez d'abord l'entraînement:
        ```
        python script.py train
        ```
        2. Les prédictions seront sauvegardées dans 'predictions_quantile/'
        """)

# ============================================================================
# CONTENU PRINCIPAL
# ============================================================================

if df is not None:
    
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
        st.header("🎯 Analyse de la prédiction précise (Quantile 50%)")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Scatter plot: Réel vs Prédit
            fig_scatter = px.scatter(
                df_sample,
                x='prix_reel',
                y='prix_predit_precis',
                title="🎯 Prédiction précise vs Prix réel",
                labels={
                    'prix_reel': 'Prix réel ($)',
                    'prix_predit_precis': 'Prédiction précise ($)'
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
            
            # Tendance
            try:
                z = np.polyfit(df_sample['prix_reel'], df_sample['prix_predit_precis'], 1)
                p = np.poly1d(z)
                
                fig_scatter.add_trace(
                    go.Scatter(
                        x=df_sample['prix_reel'].sort_values(),
                        y=p(df_sample['prix_reel'].sort_values()),
                        mode='lines',
                        name='Tendance',
                        line=dict(color='green', width=2),
                        showlegend=True
                    )
                )
            except:
                pass
            
            fig_scatter.update_layout(
                height=600,
                xaxis_title="Prix réel",
                yaxis_title="Prix prédit",
                hovermode='closest',
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Métriques de performance
            mae = df_sample['erreur'].abs().mean()
            mape = df_sample['erreur_pourcentage'].abs().mean()
            rmse = np.sqrt((df_sample['erreur'] ** 2).mean())
            r2 = df_sample['prix_reel'].corr(df_sample['prix_predit_precis']) ** 2
            
            st.markdown("""
            <div class="precision-card">
                <h3 style="color: white; margin-top: 0;">📊 Performance globale</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("MAE", f"${mae:,.0f}")
                st.metric("MAPE", f"{mape:.1f}%")
            with col2_2:
                st.metric("RMSE", f"${rmse:,.0f}")
                st.metric("R²", f"{r2:.3f}")
            
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
            fig_hist.add_vline(x=mae, line_dash="dot", line_color="green", 
                              annotation_text=f"MAE: ${mae:,.0f}")
            fig_hist.add_vline(x=-mae, line_dash="dot", line_color="green")
            
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Q-Q plot simplifié
            try:
                percentiles = np.percentile(df_sample['erreur'], np.arange(0, 101, 10))
                fig_qq = px.line(
                    x=np.arange(0, 101, 10),
                    y=percentiles,
                    title="Percentiles des erreurs",
                    labels={'x': 'Percentile', 'y': 'Erreur ($)'},
                    markers=True
                )
                fig_qq.update_layout(height=250)
                st.plotly_chart(fig_qq, use_container_width=True)
            except:
                st.info("Impossible de générer le graphique des percentiles")
    
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
            viz_sample = df_sample.head(50).reset_index()
            
            # Prix réel
            fig_interval.add_trace(go.Scatter(
                x=viz_sample.index,
                y=viz_sample['prix_reel'],
                mode='markers',
                name='Prix réel',
                marker=dict(color='red', size=8, symbol='circle'),
                hovertemplate='Prix réel: $%{y:,.0f}<br>Index: %{x}<extra></extra>'
            ))
            
            # Prédiction précise
            fig_interval.add_trace(go.Scatter(
                x=viz_sample.index,
                y=viz_sample['prix_predit_precis'],
                mode='markers+lines',
                name='Prédiction précise',
                marker=dict(color='blue', size=6),
                line=dict(color='blue', width=1),
                hovertemplate='Prédiction: $%{y:,.0f}<br>Index: %{x}<extra></extra>'
            ))
            
            # Intervalle 90%
            fig_interval.add_trace(go.Scatter(
                x=viz_sample.index,
                y=viz_sample['borne_superieure_90'],
                mode='lines',
                name='Borne supérieure (95%)',
                line=dict(dash='dash', color='green', width=1),
                hovertemplate='Borne sup: $%{y:,.0f}<br>Index: %{x}<extra></extra>'
            ))
            
            fig_interval.add_trace(go.Scatter(
                x=viz_sample.index,
                y=viz_sample['borne_inferieure_90'],
                mode='lines',
                name='Borne inférieure (5%)',
                line=dict(dash='dash', color='green', width=1),
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.1)',
                hovertemplate='Borne inf: $%{y:,.0f}<br>Index: %{x}<extra></extra>'
            ))
            
            fig_interval.update_layout(
                title=f"Intervalle de confiance 90% - Échantillon de {len(viz_sample)} propriétés",
                xaxis_title="Index",
                yaxis_title="Prix ($)",
                height=500,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_interval, use_container_width=True)
            
            # Métriques d'intervalle
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                st.metric(
                    "📉 Borne inférieure moyenne",
                    f"${df_sample['borne_inferieure_90'].mean():,.0f}"
                )
            with col2_2:
                st.metric(
                    "🎯 Prédiction précise moyenne",
                    f"${df_sample['prix_predit_precis'].mean():,.0f}"
                )
            with col2_3:
                st.metric(
                    "📈 Borne supérieure moyenne",
                    f"${df_sample['borne_superieure_90'].mean():,.0f}"
                )
        
        with col2:
            # Distribution des largeurs d'intervalle
            fig_width = px.histogram(
                df_sample,
                x='intervalle_confiance',
                nbins=50,
                title="Distribution de la largeur des intervalles",
                labels={'intervalle_confiance': 'Largeur de l\'intervalle ($)'},
                color_discrete_sequence=['#E91E63']
            )
            
            fig_width.add_vline(
                x=df_sample['intervalle_confiance'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Moyenne: ${df_sample['intervalle_confiance'].mean():,.0f}"
            )
            
            fig_width.update_layout(height=300)
            st.plotly_chart(fig_width, use_container_width=True)
            
            # Analyse par tranche de prix
            df_sample['tranche_prix'] = pd.cut(
                df_sample['prix_reel'],
                bins=[0, 300000, 600000, 900000, 1200000, 1500000, 2000000, np.inf],
                labels=['<300k', '300-600k', '600-900k', '900k-1.2M', '1.2-1.5M', '1.5-2M', '>2M']
            )
            
            # Couverture par tranche
            coverage_by_price = df_sample.groupby('tranche_prix', observed=True)['dans_intervalle'].mean() * 100
            
            fig_coverage = px.bar(
                x=coverage_by_price.index,
                y=coverage_by_price.values,
                title="Couverture par tranche de prix",
                labels={'x': 'Tranche de prix', 'y': 'Couverture (%)'},
                text=[f"{v:.1f}%" for v in coverage_by_price.values],
                color=coverage_by_price.values,
                color_continuous_scale='RdYlGn',
                range_color=[70, 100]
            )
            
            fig_coverage.add_hline(y=90, line_dash="dash", line_color="red", 
                                  annotation_text="Cible: 90%")
            
            fig_coverage.update_layout(height=300)
            st.plotly_chart(fig_coverage, use_container_width=True)
            
            # Largeur relative
            try:
                fig_rel_width = px.box(
                    df_sample,
                    x='tranche_prix',
                    y='largeur_relative',
                    title="Largeur relative par tranche de prix",
                    labels={'tranche_prix': 'Tranche de prix', 'largeur_relative': 'Largeur relative (%)'},
                    color='tranche_prix',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                
                fig_rel_width.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_rel_width, use_container_width=True)
            except:
                st.info("Impossible de générer le graphique des largeurs relatives")
    
    # ========================================================================
    # TAB 3: ANALYSE DES ERREURS
    # ========================================================================
    
    with tab3:
        st.header("📈 Analyse détaillée des erreurs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Erreur vs Prix réel
            fig_error_vs_price = px.scatter(
                df_sample,
                x='prix_reel',
                y='erreur',
                color='erreur_pourcentage',
                color_continuous_scale='RdYlBu',
                color_continuous_midpoint=0,
                title="Erreur de prédiction vs Prix réel",
                labels={
                    'prix_reel': 'Prix réel ($)',
                    'erreur': 'Erreur ($)',
                    'erreur_pourcentage': 'Erreur %'
                },
                hover_data=['prix_predit_precis']
            )
            
            fig_error_vs_price.add_hline(y=0, line_dash="dash", line_color="black")
            
            # Bandes d'erreur
            fig_error_vs_price.add_hrect(
                y0=-df_sample['erreur'].abs().mean(),
                y1=df_sample['erreur'].abs().mean(),
                fillcolor="green",
                opacity=0.1,
                line_width=0,
                annotation_text=f"± MAE: ${df_sample['erreur'].abs().mean():,.0f}"
            )
            
            fig_error_vs_price.update_layout(height=400)
            st.plotly_chart(fig_error_vs_price, use_container_width=True)
        
        with col2:
            # Erreur absolue par tranche
            df_sample['erreur_abs'] = df_sample['erreur'].abs()
            
            try:
                fig_error_box = px.box(
                    df_sample,
                    x='tranche_prix',
                    y='erreur_abs',
                    title="Erreur absolue par tranche de prix",
                    labels={'tranche_prix': 'Tranche de prix', 'erreur_abs': 'Erreur absolue ($)'},
                    color='tranche_prix',
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                
                fig_error_box.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_error_box, use_container_width=True)
            except:
                st.info("Impossible de générer le graphique des erreurs par tranche")
        
        # Top erreurs - CORRECTION ICI
        st.subheader("🔍 Top 10 des plus grandes erreurs")
        
        # Utiliser 'erreur' au lieu de 'erreur_abs' pour nlargest
        top_errors = df.nlargest(10, 'erreur')[['prix_reel', 'prix_predit_precis', 'erreur', 'erreur_pourcentage', 'borne_inferieure_90', 'borne_superieure_90']].copy()
        
        # Formatage
        for col in ['prix_reel', 'prix_predit_precis', 'erreur', 'borne_inferieure_90', 'borne_superieure_90']:
            top_errors[col] = top_errors[col].apply(lambda x: f"${x:,.0f}")
        
        top_errors['erreur_pourcentage'] = top_errors['erreur_pourcentage'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            top_errors,
            column_config={
                "prix_reel": "💰 Prix réel",
                "prix_predit_precis": "🎯 Prédiction",
                "erreur": "⚠️ Erreur",
                "erreur_pourcentage": "% Erreur",
                "borne_inferieure_90": "📉 Borne 5%",
                "borne_superieure_90": "📈 Borne 95%"
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Top 10 des meilleures prédictions
        st.subheader("✅ Top 10 des meilleures prédictions")
        
        best_predictions = df.nsmallest(10, 'erreur_abs')[['prix_reel', 'prix_predit_precis', 'erreur', 'erreur_pourcentage', 'dans_intervalle']].copy()
        
        # Formatage
        for col in ['prix_reel', 'prix_predit_precis', 'erreur']:
            best_predictions[col] = best_predictions[col].apply(lambda x: f"${x:,.0f}")
        
        best_predictions['erreur_pourcentage'] = best_predictions['erreur_pourcentage'].apply(lambda x: f"{x:.1f}%")
        best_predictions['dans_intervalle'] = best_predictions['dans_intervalle'].apply(lambda x: "✅ Oui" if x else "❌ Non")
        
        st.dataframe(
            best_predictions,
            column_config={
                "prix_reel": "💰 Prix réel",
                "prix_predit_precis": "🎯 Prédiction",
                "erreur": "⚠️ Erreur",
                "erreur_pourcentage": "% Erreur",
                "dans_intervalle": "✅ Dans intervalle"
            },
            use_container_width=True,
            hide_index=True
        )
    
    # ========================================================================
    # TAB 4: DONNÉES BRUTES
    # ========================================================================
    
    with tab4:
        st.header("📁 Visualisation des données brutes")
        
        # Filtres supplémentaires
        col1, col2, col3 = st.columns(3)
        
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
            min_error = float(df['erreur_pourcentage'].min())
            max_error = float(df['erreur_pourcentage'].max())
            error_range = st.slider(
                "Filtrer par erreur %",
                min_value=min_error,
                max_value=max_error,
                value=(min_error, max_error)
            )
        
        with col3:
            interval_filter = st.selectbox(
                "Filtrer par intervalle",
                ["Tous", "Dans l'intervalle", "Hors intervalle"]
            )
        
        # Application des filtres
        df_filtered = df[
            (df['prix_reel'] >= price_range[0]) &
            (df['prix_reel'] <= price_range[1]) &
            (df['erreur_pourcentage'] >= error_range[0]) &
            (df['erreur_pourcentage'] <= error_range[1])
        ].copy()
        
        if interval_filter == "Dans l'intervalle":
            df_filtered = df_filtered[df_filtered['dans_intervalle'] == True]
        elif interval_filter == "Hors intervalle":
            df_filtered = df_filtered[df_filtered['dans_intervalle'] == False]
        
        # Aperçu
        st.subheader(f"📊 Aperçu des données ({len(df_filtered):,} lignes)")
        
        # Colonnes à afficher
        display_cols = ['prix_reel', 'prix_predit_precis', 'borne_inferieure_90', 
                       'borne_superieure_90', 'intervalle_confiance', 'erreur', 
                       'erreur_pourcentage', 'dans_intervalle']
        
        df_display = df_filtered[display_cols].head(100).copy()
        
        # Formatage
        for col in ['prix_reel', 'prix_predit_precis', 'borne_inferieure_90', 
                   'borne_superieure_90', 'intervalle_confiance', 'erreur']:
            df_display[col] = df_display[col].apply(lambda x: f"${x:,.0f}")
        
        df_display['erreur_pourcentage'] = df_display['erreur_pourcentage'].apply(lambda x: f"{x:.1f}%")
        df_display['dans_intervalle'] = df_display['dans_intervalle'].apply(lambda x: "✅ Oui" if x else "❌ Non")
        
        st.dataframe(
            df_display,
            column_config={
                "prix_reel": "💰 Prix réel",
                "prix_predit_precis": "🎯 Prédiction",
                "borne_inferieure_90": "📉 Borne 5%",
                "borne_superieure_90": "📈 Borne 95%",
                "intervalle_confiance": "📊 Largeur",
                "erreur": "⚠️ Erreur",
                "erreur_pourcentage": "% Erreur",
                "dans_intervalle": "✅ Dans intervalle"
            },
            use_container_width=True
        )
        
        # Statistiques
        st.subheader("📊 Statistiques descriptives")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Prédictions**")
            stats_pred = df_filtered[['prix_reel', 'prix_predit_precis']].describe()
            stats_pred.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
            for col in stats_pred.columns:
                stats_pred[col] = stats_pred[col].apply(lambda x: f"${x:,.0f}" if col in ['prix_reel', 'prix_predit_precis'] else f"{x:.0f}")
            st.dataframe(stats_pred, use_container_width=True)
        
        with col2:
            st.markdown("**📊 Intervalles**")
            stats_interval = df_filtered[['borne_inferieure_90', 'borne_superieure_90', 'intervalle_confiance']].describe()
            stats_interval.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
            for col in stats_interval.columns:
                stats_interval[col] = stats_interval[col].apply(lambda x: f"${x:,.0f}")
            st.dataframe(stats_interval, use_container_width=True)
        
        # Export
        st.subheader("💾 Export des données filtrées")
        
        col1, col2 = st.columns(2)
        with col1:
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger en CSV",
                data=csv,
                file_name=f"predictions_filtrees_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            st.info(f"{len(df_filtered):,} lignes • {df_filtered.shape[1]} colonnes")

else:
    # Message d'erreur si pas de données
    st.error("❌ Aucune donnée de prédiction trouvée!")
    
    st.markdown("""
    ### 📋 Instructions
    
    1. **Entraînez d'abord le modèle** avec la commande:
    ```bash
    python script.py train
    ```
    
    2. **Vérifiez que le dossier `predictions_quantile/` contient des fichiers CSV**
    
    3. **Rafraîchissez cette page** après l'entraînement
    
    ---
    
    📁 **Structure attendue:**
    ```
    predictions_quantile/
    └── prediction_precise_intervalle90_*.csv
    ```
    """)
    
    if st.button("🔄 Vérifier à nouveau"):
        st.cache_data.clear()
        st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p style='font-size: 0.9rem;'>
            📊 <strong>Visualisation des prédictions - Intervalles de confiance 90%</strong><br>
            Quantiles: 5% (borne inférieure) | 50% (prédiction précise) | 95% (borne supérieure)<br>
            ⚡ Données chargées depuis le dossier 'predictions_quantile/'<br>
            ✅ Version corrigée - Erreur 'erreur_abs' résolue
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# UTILISATION
# ============================================================================

"""
UTILISATION:
------------
1. Visualisation simple:
   streamlit run visualisation_simple.py

2. Sans argument - charge automatiquement le dernier fichier de prédictions

CORRECTIONS APPLIQUÉES:
-----------------------
✓ Ajout de 'erreur_abs' et 'largeur_relative' dans load_latest_predictions()
✓ Correction du top_errors - utilise 'erreur' au lieu de 'erreur_abs'
✓ Ajout du top 10 des meilleures prédictions
✓ Gestion des erreurs avec try/except
✓ Formatage amélioré des données
"""