"""
metrics.py - Métriques d'évaluation pour modèles de prédiction immobilière
Calcule et exporte:
- Coverage rate (taux de couverture des intervalles)
- Average interval width (largeur moyenne des intervalles)
- Pinball Loss (perte pour les quantiles)
- MAE, RMSE, MAPE, R², Bias
- Export CSV/JSON automatique
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ModelMetrics:
    """
    Calculateur de métriques pour évaluation des modèles de prédiction
    """
    
    def __init__(self, output_dir='metrics_reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def calculate_coverage_rate(self, y_true, y_lower, y_upper):
        """
        Calcule le taux de couverture des intervalles de confiance
        """
        # Taux de couverture global
        coverage = (y_true >= y_lower) & (y_true <= y_upper)
        coverage_rate = coverage.mean() * 100
        
        # Couverture par tranche de prix
        bins = [0, 300000, 600000, 900000, 1200000, 1500000, 2000000, np.inf]
        labels = ['0-300k', '300-600k', '600-900k', '900k-1.2M', '1.2-1.5M', '1.5-2M', '2M+']
        
        y_true_series = pd.Series(y_true)
        price_bins = pd.cut(y_true_series, bins=bins, labels=labels)
        
        coverage_by_bin = {}
        for bin_label in labels:
            mask = price_bins == bin_label
            if mask.sum() > 0:
                bin_coverage = coverage[mask].mean() * 100
                coverage_by_bin[bin_label] = {
                    'coverage_rate': round(bin_coverage, 2),
                    'count': int(mask.sum()),
                    'mean_price': float(y_true[mask].mean())
                }
        
        logger.info(f"📊 Coverage Global: {coverage_rate:.1f}%")
        
        return {
            'global_coverage_rate': round(coverage_rate, 2),
            'coverage_by_bin': coverage_by_bin
        }
    
    def calculate_interval_width(self, y_lower, y_upper):
        """
        Calcule la largeur moyenne des intervalles
        """
        widths = y_upper - y_lower
        
        stats = {
            'mean_width': float(np.mean(widths)),
            'median_width': float(np.median(widths)),
            'std_width': float(np.std(widths)),
            'min_width': float(np.min(widths)),
            'max_width': float(np.max(widths))
        }
        
        logger.info(f"📏 Largeur moyenne: ${stats['mean_width']:,.0f}")
        
        return stats
    
    def calculate_pinball_loss(self, y_true, y_pred, quantile):
        """
        Calcule la Pinball Loss pour un quantile donné
        """
        errors = y_true - y_pred
        pinball = np.maximum(quantile * errors, (quantile - 1) * errors)
        mean_pinball = np.mean(pinball)
        
        logger.info(f"🎯 Pinball Loss (q={quantile}): {mean_pinball:.2f}")
        
        return round(mean_pinball, 2)
    
    def calculate_regression_metrics(self, y_true, y_pred):
        """
        Calcule toutes les métriques de régression standard
        """
        metrics = {}
        
        # Métriques principales
        metrics['mae'] = round(mean_absolute_error(y_true, y_pred), 2)
        metrics['rmse'] = round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
        metrics['r2'] = round(r2_score(y_true, y_pred), 4)
        
        # MAPE
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        metrics['mape'] = round(mape, 2)
        
        # Bias et erreurs
        errors = y_true - y_pred
        metrics['bias'] = round(np.mean(errors), 2)
        metrics['std_errors'] = round(np.std(errors), 2)
        metrics['median_ae'] = round(np.median(np.abs(errors)), 2)
        metrics['max_error'] = round(np.max(np.abs(errors)), 2)
        
        logger.info(f"📈 MAE: ${metrics['mae']:,.0f} | R²: {metrics['r2']:.4f} | MAPE: {metrics['mape']:.1f}%")
        
        return metrics
    
    def calculate_all_metrics(self, y_true, y_pred_median, y_pred_lower, y_pred_upper, model_name='model'):
        """
        Calcule toutes les métriques pour un modèle avec intervalles
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"MÉTRIQUES - {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        all_metrics = {
            'model_name': model_name,
            'timestamp': self.timestamp,
            'n_samples': len(y_true)
        }
        
        # 1. Métriques de régression
        regression_metrics = self.calculate_regression_metrics(y_true, y_pred_median)
        all_metrics.update(regression_metrics)
        
        # 2. Coverage Rate
        coverage_results = self.calculate_coverage_rate(y_true, y_pred_lower, y_pred_upper)
        all_metrics['coverage_rate'] = coverage_results['global_coverage_rate']
        
        # 3. Largeur des intervalles
        width_stats = self.calculate_interval_width(y_pred_lower, y_pred_upper)
        all_metrics['interval_mean_width'] = width_stats['mean_width']
        all_metrics['interval_median_width'] = width_stats['median_width']
        
        # 4. Pinball Loss
        all_metrics['pinball_q05'] = self.calculate_pinball_loss(y_true, y_pred_lower, 0.05)
        all_metrics['pinball_q50'] = self.calculate_pinball_loss(y_true, y_pred_median, 0.50)
        all_metrics['pinball_q95'] = self.calculate_pinball_loss(y_true, y_pred_upper, 0.95)
        
        return all_metrics
    
    def export_to_csv(self, metrics_dict, filename=None):
        """
        Exporte les métriques au format CSV
        """
        if filename is None:
            model_name = metrics_dict.get('model_name', 'model')
            filename = f"metrics_{model_name}_{self.timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Convertir en DataFrame
        df_metrics = pd.DataFrame([metrics_dict])
        
        # Arrondir les valeurs
        for col in df_metrics.select_dtypes(include=[np.number]).columns:
            if 'mae' in col or 'rmse' in col or 'width' in col or 'bias' in col:
                df_metrics[col] = df_metrics[col].round(0)
            elif 'r2' in col or 'coverage' in col or 'pinball' in col:
                df_metrics[col] = df_metrics[col].round(4)
        
        df_metrics.to_csv(filepath, index=False)
        logger.info(f"✅ CSV sauvegardé: {filepath}")
        
        return filepath
    
    def export_coverage_details(self, coverage_results, model_name='model'):
        """
        Exporte les détails de couverture par tranche
        """
        coverage_data = []
        
        for bin_label, stats in coverage_results['coverage_by_bin'].items():
            coverage_data.append({
                'price_range': bin_label,
                'coverage_rate': stats['coverage_rate'],
                'count': stats['count'],
                'mean_price': f"${stats['mean_price']:,.0f}"
            })
        
        df_coverage = pd.DataFrame(coverage_data)
        
        filename = f"coverage_{model_name}_{self.timestamp}.csv"
        filepath = self.output_dir / filename
        df_coverage.to_csv(filepath, index=False)
        
        logger.info(f"✅ Couverture exportée: {filepath}")
        
        return filepath
    
    def export_all_metrics(self, y_true, predictions_dict, y_lower=None, y_upper=None):
        """
        Exporte toutes les métriques pour tous les modèles
        """
        exported_files = {}
        all_models_metrics = {}
        
        for model_name, y_pred in predictions_dict.items():
            # Vérifier si on a des intervalles
            has_interval = (y_lower is not None and 
                          model_name in y_lower and 
                          model_name in y_upper)
            
            if has_interval:
                metrics = self.calculate_all_metrics(
                    y_true, y_pred, 
                    y_lower[model_name], 
                    y_upper[model_name],
                    model_name
                )
                
                # Export détails couverture
                coverage_results = self.calculate_coverage_rate(
                    y_true, y_lower[model_name], y_upper[model_name]
                )
                coverage_file = self.export_coverage_details(coverage_results, model_name)
                exported_files[f'coverage_{model_name}'] = coverage_file
                
            else:
                # Métriques de régression seulement
                metrics = self.calculate_regression_metrics(y_true, y_pred)
                metrics['model_name'] = model_name
                metrics['timestamp'] = self.timestamp
                metrics['n_samples'] = len(y_true)
            
            all_models_metrics[model_name] = metrics
            
            # Export CSV
            csv_file = self.export_to_csv(metrics, f"metrics_{model_name}_{self.timestamp}.csv")
            exported_files[f'metrics_{model_name}'] = csv_file
        
        # Comparaison des modèles
        if len(all_models_metrics) > 1:
            comparison_data = []
            for model_name, metrics in all_models_metrics.items():
                comparison_data.append({
                    'Modèle': model_name,
                    'MAE ($)': f"{metrics.get('mae', 0):,.0f}",
                    'RMSE ($)': f"{metrics.get('rmse', 0):,.0f}",
                    'R²': f"{metrics.get('r2', 0):.4f}",
                    'MAPE (%)': f"{metrics.get('mape', 0):.1f}",
                    'Coverage (%)': f"{metrics.get('coverage_rate', 0):.1f}",
                    'Bias ($)': f"{metrics.get('bias', 0):,.0f}"
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            comparison_file = self.output_dir / f"comparison_{self.timestamp}.csv"
            df_comparison.to_csv(comparison_file, index=False)
            exported_files['comparison'] = comparison_file
            
            logger.info(f"\n📊 COMPARAISON DES MODÈLES:")
            logger.info(df_comparison.to_string(index=False))
        
        # Résumé JSON
        summary = {
            'timestamp': self.timestamp,
            'n_models': len(all_models_metrics),
            'models': list(all_models_metrics.keys()),
            'metrics': all_models_metrics
        }
        
        json_file = self.output_dir / f"summary_{self.timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        exported_files['summary'] = json_file
        
        logger.info(f"\n✅ {len(exported_files)} fichiers exportés dans {self.output_dir}/")
        
        return exported_files


# ============================================================================
# FONCTIONS RAPIDES POUR UTILISATION DIRECTE
# ============================================================================

def quick_evaluate(y_true, y_pred, model_name='model'):
    """
    Évaluation rapide d'un modèle de régression
    """
    metrics = ModelMetrics()
    
    # Calcul des métriques
    results = metrics.calculate_regression_metrics(y_true, y_pred)
    results['model_name'] = model_name
    
    # Export
    df = pd.DataFrame([results])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"quick_metrics_{model_name}_{timestamp}.csv"
    filepath = metrics.output_dir / filename
    df.to_csv(filepath, index=False)
    
    # Affichage
    print(f"\n📊 {model_name}:")
    print(f"   MAE : ${results['mae']:,.0f}")
    print(f"   RMSE: ${results['rmse']:,.0f}")
    print(f"   R²  : {results['r2']:.4f}")
    print(f"   MAPE: {results['mape']:.1f}%")
    print(f"✅ Fichier: {filepath}")
    
    return results

def evaluate_with_intervals(y_true, y_median, y_lower, y_upper, model_name='quantile_model'):
    """
    Évaluation complète avec intervalles de confiance
    """
    metrics = ModelMetrics()
    
    # Calcul des métriques complètes
    results = metrics.calculate_all_metrics(
        y_true, y_median, y_lower, y_upper, model_name
    )
    
    # Export
    csv_file = metrics.export_to_csv(results)
    
    coverage_results = metrics.calculate_coverage_rate(y_true, y_lower, y_upper)
    coverage_file = metrics.export_coverage_details(coverage_results, model_name)
    
    # Affichage
    print(f"\n📊 {model_name.upper()} - RAPPORT COMPLET:")
    print(f"   MAE : ${results['mae']:,.0f} ({results['mape']:.1f}%)")
    print(f"   R²  : {results['r2']:.4f}")
    print(f"   Coverage 90%: {results['coverage_rate']:.1f}%")
    print(f"   Largeur intervalle: ${results['interval_mean_width']:,.0f}")
    print(f"✅ CSV: {csv_file}")
    print(f"✅ Couverture: {coverage_file}")
    
    return results

def compare_models(models_predictions, y_true):
    """
    Compare plusieurs modèles
    """
    metrics = ModelMetrics()
    
    all_metrics = {}
    for model_name, y_pred in models_predictions.items():
        all_metrics[model_name] = metrics.calculate_regression_metrics(y_true, y_pred)
        
        # Export individuel
        df = pd.DataFrame([all_metrics[model_name]])
        filename = f"metrics_{model_name}_{metrics.timestamp}.csv"
        df.to_csv(metrics.output_dir / filename, index=False)
    
    # Tableau comparatif
    comparison_data = []
    for model_name, m in all_metrics.items():
        comparison_data.append({
            'Modèle': model_name,
            'MAE': f"${m['mae']:,.0f}",
            'RMSE': f"${m['rmse']:,.0f}",
            'R²': f"{m['r2']:.4f}",
            'MAPE': f"{m['mape']:.1f}%",
            'Bias': f"${m['bias']:,.0f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    comparison_file = metrics.output_dir / f"comparison_{metrics.timestamp}.csv"
    df_comparison.to_csv(comparison_file, index=False)
    
    print(f"\n📊 COMPARAISON DES MODÈLES:")
    print(df_comparison.to_string(index=False))
    print(f"\n✅ Fichier: {comparison_file}")
    
    return df_comparison


# ============================================================================
# TEST DU MODULE
# ============================================================================

if __name__ == "__main__":
    """
    Exemple d'utilisation
    """
    print("\n" + "="*70)
    print("📊 MODULE DE MÉTRIQUES - TEST")
    print("="*70)
    
    # Données simulées
    np.random.seed(42)
    n = 1000
    y_true = np.random.normal(500000, 150000, n)
    y_pred = y_true + np.random.normal(0, 25000, n)
    y_lower = y_pred - np.random.normal(70000, 10000, n)
    y_upper = y_pred + np.random.normal(70000, 10000, n)
    
    # 1. Évaluation rapide
    print("\n🔹 TEST 1: Évaluation rapide")
    quick_evaluate(y_true, y_pred, "XGBoost")
    
    # 2. Évaluation avec intervalles
    print("\n🔹 TEST 2: Évaluation avec intervalles")
    evaluate_with_intervals(y_true, y_pred, y_lower, y_upper, "LightGBM")
    
    # 3. Comparaison
    print("\n🔹 TEST 3: Comparaison de modèles")
    models = {
        "XGBoost": y_pred,
        "LightGBM": y_pred + np.random.normal(0, 5000, n),
        "RandomForest": y_pred - np.random.normal(0, 3000, n)
    }
    compare_models(models, y_true)
    
    print(f"\n✅ Fichiers générés dans /{ModelMetrics().output_dir}/")