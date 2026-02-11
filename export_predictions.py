#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour exporter les prédictions du modèle au format CSV
"""

import pandas as pd
import numpy as np
from model import load_model, prepare_data
import os
from datetime import datetime

def export_predictions_csv(data_path='washington_real_estate.csv', 
                          model_path='models/washington_real_estate_model.pkl',
                          output_dir='exports'):
    """
    Exporte les prédictions au format CSV uniquement
    """
    
    print("=" * 60)
    print("EXPORTATION DES PRÉDICTIONS - FORMAT CSV")
    print("=" * 60)
    
    # Créer le dossier d'export
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Dossier créé: {output_dir}")
    
    # 1. EXPORT DES PRÉDICTIONS COMPLÈTES
    print("\n📁 1. Export des prédictions complètes...")
    
    # Charger le modèle
    model = load_model(model_path)
    if model is None:
        print("✗ Modèle non trouvé!")
        return
    
    # Charger les données
    df = pd.read_csv(data_path)
    print(f"✓ Données chargées: {len(df)} enregistrements")
    
    # Préparer les features
    X, y = prepare_data(df)
    
    # Prédictions
    y_pred = model.predict(X)
    
    # Créer DataFrame complet
    results = pd.DataFrame()
    
    # Identifiants
    results['property_id'] = range(1, len(df) + 1)
    results['city'] = df['city']
    results['sale_date'] = df['sale_date']
    
    # Caractéristiques
    results['sqft'] = df['sqft']
    results['sqft_lot'] = df['sqft_lot']
    results['beds'] = df['beds']
    results['bath_full'] = df['bath_full']
    results['bath_3qtr'] = df['bath_3qtr']
    results['bath_half'] = df['bath_half']
    results['grade'] = df['grade']
    results['condition'] = df['condition']
    results['year_built'] = df['year_built']
    
    # Prix
    results['actual_price'] = y.values
    results['predicted_price'] = y_pred
    results['price_difference'] = y_pred - y.values
    results['abs_difference'] = np.abs(y_pred - y.values)
    results['error_percentage'] = (np.abs(y_pred - y.values) / y.values) * 100
    
    # Arrondir
    price_cols = ['actual_price', 'predicted_price', 'price_difference', 'abs_difference']
    results[price_cols] = results[price_cols].round(0)
    results['error_percentage'] = results['error_percentage'].round(2)
    
    # Classification
    results['prediction_quality'] = pd.cut(
        results['error_percentage'],
        bins=[0, 5, 10, 20, 100],
        labels=['Excellent', 'Bon', 'Moyen', 'Faible']
    )
    
    # Sauvegarde CSV principal
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_csv = f'{output_dir}/predictions_completes_{timestamp}.csv'
    results.to_csv(main_csv, index=False, encoding='utf-8-sig')
    print(f"  ✓ {main_csv}")
    print(f"  • {len(results):,} lignes")
    
    # 2. EXPORT PAR VILLE (CSV séparés)
    print("\n📁 2. Export par ville...")
    
    city_dir = f'{output_dir}/par_ville'
    if not os.path.exists(city_dir):
        os.makedirs(city_dir)
    
    for city in results['city'].unique():
        city_data = results[results['city'] == city]
        city_file = f'{city_dir}/{city.lower()}_predictions_{timestamp}.csv'
        city_data.to_csv(city_file, index=False, encoding='utf-8-sig')
    
    print(f"  ✓ {len(results['city'].unique())} fichiers CSV créés")
    
    # 3. EXPORT RÉSUMÉ STATISTIQUE
    print("\n📁 3. Export des statistiques...")
    
    # Statistiques globales
    summary_data = {
        'Date_export': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Total_proprietes': len(results),
        'Prix_moyen_reel': results['actual_price'].mean(),
        'Prix_moyen_prediction': results['predicted_price'].mean(),
        'Erreur_moyenne': results['abs_difference'].mean(),
        'Erreur_mediane': results['abs_difference'].median(),
        'Erreur_pourcentage_moyen': results['error_percentage'].mean(),
        'RMSE': np.sqrt((results['price_difference'] ** 2).mean()),
        'R2': 1 - (results['price_difference'] ** 2).sum() / ((results['actual_price'] - results['actual_price'].mean()) ** 2).sum(),
        'Predictions_excellentes': len(results[results['prediction_quality'] == 'Excellent']),
        'Predictions_bonnes': len(results[results['prediction_quality'] == 'Bon']),
        'Predictions_moyennes': len(results[results['prediction_quality'] == 'Moyen']),
        'Predictions_faibles': len(results[results['prediction_quality'] == 'Faible'])
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_csv = f'{output_dir}/statistiques_globales_{timestamp}.csv'
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f"  ✓ {summary_csv}")
    
    # Statistiques par ville
    city_stats = results.groupby('city').agg({
        'actual_price': ['count', 'mean', 'median'],
        'predicted_price': 'mean',
        'abs_difference': 'mean',
        'error_percentage': 'mean',
        'price_difference': 'mean'
    }).round(2)
    
    city_stats.columns = ['nb_proprietes', 'prix_moyen_reel', 'prix_median_reel', 
                         'prix_moyen_prediction', 'erreur_moyenne_abs', 
                         'erreur_pct_moyen', 'biais_moyen']
    city_stats = city_stats.reset_index()
    city_stats_csv = f'{output_dir}/statistiques_par_ville_{timestamp}.csv'
    city_stats.to_csv(city_stats_csv, index=False, encoding='utf-8-sig')
    print(f"  ✓ {city_stats_csv}")
    
    # 4. EXPORT DES MEILLEURES ET PIRE PRÉDICTIONS
    print("\n📁 4. Export des cas extrêmes...")
    
    # Top 20 meilleures prédictions
    best = results.nsmallest(20, 'abs_difference')
    best_csv = f'{output_dir}/meilleures_predictions_{timestamp}.csv'
    best.to_csv(best_csv, index=False, encoding='utf-8-sig')
    print(f"  ✓ {best_csv}")
    
    # Top 20 pires prédictions
    worst = results.nlargest(20, 'abs_difference')
    worst_csv = f'{output_dir}/pires_predictions_{timestamp}.csv'
    worst.to_csv(worst_csv, index=False, encoding='utf-8-sig')
    print(f"  ✓ {worst_csv}")
    
    # 5. EXPORT POUR ANALYSE (format light)
    print("\n📁 5. Export version légère...")
    
    light_cols = ['property_id', 'city', 'sqft', 'beds', 'actual_price', 
                  'predicted_price', 'error_percentage', 'prediction_quality']
    light_df = results[light_cols]
    light_csv = f'{output_dir}/predictions_light_{timestamp}.csv'
    light_df.to_csv(light_csv, index=False, encoding='utf-8-sig')
    print(f"  ✓ {light_csv}")
    
    # 6. FICHIER README
    print("\n📁 6. Génération du README...")
    
    readme_content = f"""# EXPORT DES PRÉDICTIONS - KING COUNTY REAL ESTATE
Date d'export: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## FICHIERS GÉNÉRÉS

1. **predictions_completes_{timestamp}.csv**
   - Toutes les propriétés avec toutes les caractéristiques
   - Prix réels et prédits
   - Métriques d'erreur complètes

2. **par_ville/***
   - Fichiers séparés par ville
   - Format: [ville]_predictions_{timestamp}.csv

3. **statistiques_globales_{timestamp}.csv**
   - Résumé global des performances
   - Métriques principales

4. **statistiques_par_ville_{timestamp}.csv**
   - Performances détaillées par ville

5. **meilleures_predictions_{timestamp}.csv**
   - Top 20 des prédictions les plus précises

6. **pires_predictions_{timestamp}.csv**
   - Top 20 des prédictions les moins précises

7. **predictions_light_{timestamp}.csv**
   - Version allégée pour analyse rapide

## MÉTRIQUES GLOBALES
- Total propriétés: {len(results):,}
- Erreur moyenne: ${results['abs_difference'].mean():,.0f}
- Erreur % moyenne: {results['error_percentage'].mean():.1f}%
- Précision moyenne: {100 - results['error_percentage'].mean():.1f}%
- R²: {summary_data['R2']:.3f}

## STRUCTURE DES COLONNES

### Colonnes communes:
- property_id: Identifiant unique
- city: Ville
- sale_date: Date de vente
- sqft, sqft_lot: Superficies
- beds, baths: Chambres et salles de bain
- grade, condition: Notes de qualité
- year_built: Année de construction
- actual_price: Prix réel de vente
- predicted_price: Prix prédit
- price_difference: Différence (prédit - réel)
- abs_difference: Différence absolue
- error_percentage: Erreur en pourcentage
- prediction_quality: Qualité de la prédiction (Excellent/Bon/Moyen/Faible)

---
Généré automatiquement par le système de prédiction King County Real Estate
"""
    
    readme_file = f'{output_dir}/README_{timestamp}.txt'
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"  ✓ {readme_file}")
    
    # RÉSUMÉ FINAL
    print("\n" + "=" * 60)
    print("RÉSUMÉ DE L'EXPORTATION")
    print("=" * 60)
    print(f"\n📊 {len(results):,} prédictions exportées")
    print(f"📁 Dossier: {output_dir}/")
    print(f"\nFichiers créés:")
    print(f"  • 1 principal: predictions_completes_{timestamp}.csv")
    print(f"  • {len(results['city'].unique())} fichiers par ville")
    print(f"  • 2 fichiers de statistiques")
    print(f"  • 2 fichiers cas extrêmes")
    print(f"  • 1 version légère")
    print(f"  • 1 fichier README")
    print(f"\n✅ Export CSV terminé avec succès!")
    
    return results

def export_echantillon_csv(data_path='washington_real_estate.csv', 
                          model_path='models/washington_real_estate_model.pkl',
                          n_samples=100):
    """
    Exporte un échantillon des prédictions pour test
    """
    results = export_predictions_csv(data_path, model_path, 'exports_test')
    echantillon = results.sample(n=min(n_samples, len(results)))
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    echantillon_file = f'exports_test/echantillon_{n_samples}_{timestamp}.csv'
    echantillon.to_csv(echantillon_file, index=False, encoding='utf-8-sig')
    
    print(f"\n📋 Échantillon de {len(echantillon)} prédictions: {echantillon_file}")
    return echantillon

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("EXPORTATION DES PRÉDICTIONS AU FORMAT CSV")
    print("=" * 60)
    print("\nOptions disponibles:")
    print("  1. Export complet (toutes les prédictions)")
    print("  2. Export échantillon (100 lignes)")
    print("  3. Export par ville uniquement")
    
    choice = input("\nVotre choix (1/2/3) [défaut: 1]: ").strip() or "1"
    
    if choice == "1":
        results = export_predictions_csv()
    elif choice == "2":
        results = export_echantillon_csv(n_samples=100)
    elif choice == "3":
        # Export uniquement par ville
        results = export_predictions_csv()
        print("\n📁 Les fichiers par ville sont dans 'exports/par_ville/'")
    else:
        print("Choix non valide, export complet par défaut")
        results = export_predictions_csv()
    
    # Afficher les 5 premières lignes
    if results is not None:
        print("\n📋 APERÇU DES DONNÉES (5 premières lignes):")
        preview_cols = ['city', 'sqft', 'actual_price', 'predicted_price', 
                       'error_percentage', 'prediction_quality']
        print(results[preview_cols].head().to_string())