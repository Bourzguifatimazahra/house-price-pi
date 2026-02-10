import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

def prepare_data(df):
    """
    Prépare les données pour le modèle prédictif
    """
    # Colonnes disponibles
    available_cols = df.columns.tolist()
    
    # Features de base (vérifier si elles existent)
    base_features = ['city', 'sqft', 'sqft_lot', 'beds', 'bath_full', 
                    'bath_3qtr', 'bath_half', 'grade', 'condition',
                    'year_built']
    
    # Features optionnelles
    optional_features = ['stories', 'gara_sqft', 'greenbelt', 
                        'view_rainier', 'view_otherwater',
                        'latitude', 'longitude']
    
    # Construire la liste des features disponibles
    features = []
    for feature in base_features:
        if feature in available_cols:
            features.append(feature)
    
    for feature in optional_features:
        if feature in available_cols:
            features.append(feature)
    
    print(f"Features utilisées ({len(features)}): {features}")
    
    # Target
    target = 'sale_price'
    
    # Séparation
    X = df[features]
    y = df[target]
    
    return X, y

def build_model(X, y):
    """
    Construit et entraîne un modèle prédictif
    """
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Identifier les types de features
    categorical_features = []
    numerical_features = []
    binary_features = []
    
    for col in X.columns:
        if col == 'city':
            categorical_features.append(col)
        elif X[col].nunique() == 2 and set(X[col].unique()).issubset({0, 1}):
            binary_features.append(col)
        else:
            numerical_features.append(col)
    
    print(f"Categorical: {categorical_features}")
    print(f"Numerical: {numerical_features}")
    print(f"Binary: {binary_features}")
    
    # Préprocesseurs
    transformers = []
    
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    if numerical_features:
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numerical_transformer, numerical_features))
    
    if binary_features:
        transformers.append(('bin', 'passthrough', binary_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Entraînement
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    
    # Métriques
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    result = {
        'model': model,
        'metrics': {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        },
        'test_data': {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    }
    
    return result

def train_and_save_model(data_path='washington_real_estate.csv'):
    """
    Entraîne et sauvegarde le modèle
    """
    try:
        # Chargement
        df = pd.read_csv(data_path)
        print(f"✓ Données chargées: {len(df)} enregistrements")
        
        if len(df) < 100:
            raise ValueError(f"Pas assez de données: {len(df)}")
        
        # Préparation
        X, y = prepare_data(df)
        
        # Construction
        print("Entraînement du modèle...")
        result = build_model(X, y)
        
        # Sauvegarde
        if not os.path.exists('models'):
            os.makedirs('models')
        
        with open('models/washington_real_estate_model.pkl', 'wb') as f:
            pickle.dump(result['model'], f)
        
        with open('models/model_metrics.pkl', 'wb') as f:
            pickle.dump(result['metrics'], f)
        
        print("✓ Modèle sauvegardé")
        return result
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return None

def load_model(model_path='models/washington_real_estate_model.pkl'):
    """
    Charge le modèle sauvegardé
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"✗ Erreur chargement: {e}")
        return None

def predict_price(model, input_data):
    """
    Prédit le prix d'une propriété
    """
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data])
    
    prediction = model.predict(input_data)[0]
    return prediction

if __name__ == "__main__":
    print("=" * 50)
    print("ENTRAÎNEMENT DU MODÈLE")
    print("=" * 50)
    
    result = train_and_save_model()
    
    if result:
        print("\nRÉSULTATS:")
        print("-" * 30)
        for metric, value in result['metrics'].items():
            if metric != 'R2':
                print(f"{metric}: ${value:,.2f}")
            else:
                print(f"{metric}: {value:.3f}")
        
        print(f"\nInterprétation:")
        print(f"• Erreur moyenne: ${result['metrics']['MAE']:,.0f}")
        print(f"• Précision: {result['metrics']['R2']*100:.1f}%")
        print("\n✓ Prêt à l'emploi")
    else:
        print("\n✗ Échec de l'entraînement")