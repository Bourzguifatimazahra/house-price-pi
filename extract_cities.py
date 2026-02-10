import pandas as pd

def extract_unique_cities(file_path):
    try:
        # Charger le dataset
        df = pd.read_csv(file_path)

        # Vérifier si la colonne 'city' existe
        if 'city' in df.columns:
            # Obtenir les noms de villes uniques
            unique_cities = df['city'].unique()
            print("Voici la liste des villes uniques trouvées dans le fichier :")
            for city in sorted(unique_cities):
                print(city)
        else:
            print("La colonne 'city' n'a pas été trouvée dans le fichier CSV.")

    except FileNotFoundError:
        print(f"Le fichier {file_path} n'a pas été trouvé.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    # Chemin vers votre fichier CSV
    csv_file = 'dataset.csv'
    extract_unique_cities(csv_file)