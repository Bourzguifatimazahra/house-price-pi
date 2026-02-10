import pandas as pd
import numpy as np
from datetime import datetime

def generate_washington_real_estate_data(num_records=1000):
    """
    Génère des données synthétiques pour le marché immobilier de King County, Washington
    """
    # Toutes les villes de King County
    cities = [
        'ALGONA', 'AUBURN', 'BEAUX ARTS', 'BELLEVUE', 'BLACK DIAMOND',
        'BOTHELL', 'BURIEN', 'CARNATION', 'CLYDE HILL', 'COVINGTON',
        'DES MOINES', 'DUVALL', 'ENUMCLAW', 'FEDERAL WAY', 'HUNTS POINT',
        'ISSAQUAH', 'KENMORE', 'KENT', 'KING COUNTY', 'KIRKLAND',
        'LAKE FOREST PARK', 'MAPLE VALLEY', 'MEDINA', 'MERCER ISLAND',
        'MILTON', 'NEWCASTLE', 'NORMANDY PARK', 'NORTH BEND', 'PACIFIC',
        'REDMOND', 'RENTON', 'SAMMAMISH', 'SEA-TAC', 'SEATTLE',
        'SHORELINE', 'SKYKOMISH', 'SNOQUALMIE', 'SeaTac', 'TUKWILA',
        'WOODINVILLE', 'YARROW POINT'
    ]
    
    # Prix moyens par ville (USD)
    price_stats = {
        'SEATTLE': (850000, 350000),
        'BELLEVUE': (1200000, 400000),
        'REDMOND': (1000000, 250000),
        'KIRKLAND': (950000, 220000),
        'RENTON': (650000, 130000),
        'SAMMAMISH': (1300000, 300000),
        'MERCER ISLAND': (2200000, 450000),
        'MEDINA': (3500000, 700000),
        'HUNTS POINT': (3000000, 600000),
        'YARROW POINT': (2800000, 550000),
        'CLYDE HILL': (2500000, 500000),
        'BEAUX ARTS': (1500000, 300000),
        'BOTHELL': (750000, 150000),
        'SHORELINE': (750000, 150000),
        'NEWCASTLE': (950000, 200000),
        'ISSAQUAH': (900000, 200000),
        'WOODINVILLE': (900000, 190000),
        'KENMORE': (800000, 160000),
        'SNOQUALMIE': (850000, 170000),
        'MAPLE VALLEY': (600000, 120000),
        'COVINGTON': (550000, 110000),
        'BURIEN': (600000, 120000),
        'DES MOINES': (580000, 120000),
        'TUKWILA': (520000, 105000),
        'FEDERAL WAY': (500000, 100000),
        'KENT': (550000, 110000),
        'AUBURN': (500000, 100000),
        'ENUMCLAW': (520000, 100000),
        'NORTH BEND': (700000, 140000),
        'DUVALL': (700000, 140000),
        'CARNATION': (650000, 130000),
        'ALGONA': (450000, 80000),
        'PACIFIC': (450000, 80000),
        'MILTON': (480000, 90000),
        'SKYKOMISH': (400000, 80000),
        'LAKE FOREST PARK': (850000, 180000),
        'NORMANDY PARK': (1100000, 250000),
        'BLACK DIAMOND': (550000, 120000),
        'SEA-TAC': (550000, 110000),
        'SeaTac': (560000, 115000),
        'KING COUNTY': (600000, 150000)
    }
    
    # Subdivisions possibles
    subdivisions = [
        'ALDERWOOD SOUTH DIV NO. 02',
        'WILDWOOD LANE NO. 03', 
        'FALCON RIDGE (CEDAR RIDGE)',
        'OLYMPIC VUE ESTATES',
        'HOLLYWOOD HILL HIGHLANDS',
        'TOWN-COUNTRY CLUB BUNGALOW SITES',
        'CEDAR HEIGHTS',
        'MAPLE GROVE',
        'PINE VIEW ESTATES',
        'RIVERBEND TERRACE',
        'SUNSET HILLS',
        'MOUNTAIN VIEW',
        'LAKEFRONT VILLAGE',
        'HIGHLAND PARK',
        'GREEN VALLEY',
        'FOREST GLEN',
        'MEADOWBROOK',
        'OAKWOOD MANOR',
        'SUMMIT RIDGE',
        'BAYVIEW HEIGHTS'
    ]
    
    # Zonings
    zonings = ['R-1', 'R-2', 'R-3', 'R-4', 'R-5', 'R-6', 'R-7', 'R-8', 'R-9', 'R-10',
               'RS-7200', 'RS7.2', 'RA2.5', 'R-8', 'RS 8.5']
    
    # Submarkets
    submarkets = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    data = []
    current_year = 2024
    
    for i in range(num_records):
        # Ville aléatoire
        city = np.random.choice(cities)
        price_mean, price_std = price_stats.get(city, (600000, 150000))
        
        # Prix de base
        base_price = max(100000, np.random.normal(price_mean, price_std))
        
        # Caractéristiques principales
        sqft = np.random.randint(800, 5000)
        sqft_lot = np.random.randint(2000, 30000)
        beds = np.random.randint(1, 6)
        bath_full = np.random.randint(1, 4)
        bath_3qtr = np.random.randint(0, 2)
        bath_half = np.random.randint(0, 2)
        grade = np.random.randint(3, 13)
        condition = np.random.randint(1, 6)
        year_built = np.random.randint(1900, 2023)
        stories = np.random.randint(1, 3)
        
        # Ajustements de prix
        price = base_price
        price *= 1 + (grade - 7) * 0.05
        price *= 1 - (2023 - year_built) * 0.002
        price = int(max(50000, price))
        
        # Valeurs terrain/améliorations
        land_val = int(price * np.random.uniform(0.3, 0.7))
        imp_val = price - land_val
        
        # Dates
        sale_year = np.random.randint(1990, 2024)
        sale_month = np.random.randint(1, 13)
        sale_day = 15
        
        # Coordonnées
        if city == 'SEATTLE':
            lat, lng = np.random.uniform(47.5, 47.8), np.random.uniform(-122.4, -122.2)
        elif city == 'BELLEVUE':
            lat, lng = np.random.uniform(47.55, 47.65), np.random.uniform(-122.2, -122.1)
        elif city == 'REDMOND':
            lat, lng = np.random.uniform(47.62, 47.7), np.random.uniform(-122.1, -122.0)
        elif city == 'KIRKLAND':
            lat, lng = np.random.uniform(47.65, 47.7), np.random.uniform(-122.25, -122.1)
        else:
            lat, lng = np.random.uniform(47.3, 47.8), np.random.uniform(-122.4, -121.8)
        
        # Création de l'entrée
        entry = {
            'id': i,
            'sale_date': f"{sale_year}-{sale_month:02d}-{sale_day:02d}",
            'sale_price': price,
            'sale_nbr': np.random.choice([1.0, 2.0, 3.0, 4.0]),
            'sale_warning': '   ',
            'join_status': 'nochg',
            'join_year': current_year,
            'latitude': round(lat, 4),
            'longitude': round(lng, 4),
            'area': np.random.randint(1, 100),
            'city': city,
            'zoning': np.random.choice(zonings),
            'subdivision': np.random.choice(subdivisions),
            'present_use': 2,
            'land_val': land_val,
            'imp_val': imp_val,
            'year_built': year_built,
            'year_reno': 0,
            'sqft_lot': sqft_lot,
            'sqft': sqft,
            'sqft_1': sqft,
            'sqft_fbsmt': 0,
            'grade': grade,
            'fbsmt_grade': 0,
            'condition': condition,
            'stories': float(stories),
            'beds': beds,
            'bath_full': bath_full,
            'bath_3qtr': bath_3qtr,
            'bath_half': bath_half,
            'garb_sqft': 0,
            'gara_sqft': np.random.randint(0, 600),
            'wfnt': 0,
            'golf': 0,
            'greenbelt': np.random.choice([0, 1]),
            'noise_traffic': 0,
            'view_rainier': np.random.choice([0, 1]),
            'view_olympics': 0,
            'view_cascades': 0,
            'view_territorial': 0,
            'view_skyline': 0,
            'view_sound': 0,
            'view_lakewash': 0,
            'view_lakesamm': 0,
            'view_otherwater': np.random.choice([0, 1]),
            'view_other': 0,
            'submarket': np.random.choice(submarkets)
        }
        
        data.append(entry)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Génération des données Washington...")
    df = generate_washington_real_estate_data(2000)
    df.to_csv('washington_real_estate.csv', index=False)
    
    print(f"✓ {len(df)} propriétés générées")
    print(f"✓ {df['city'].nunique()} villes")
    print(f"✓ Prix moyen: ${df['sale_price'].mean():,.0f}")
    print(f"✓ Colonnes: {list(df.columns)}")