import pandas as pd
import os

def convert_csv_to_excel(csv_file, excel_file=None):
    """
    Convertit un fichier CSV en fichier Excel (.xlsx)
    """
    if excel_file is None:
        excel_file = csv_file.replace('.csv', '.xlsx')
    
    try:
        # Lire le CSV
        df = pd.read_csv(csv_file, encoding='utf-8')
        
        # Écrire en Excel (openpyxl est requis)
        df.to_excel(excel_file, index=False, engine='openpyxl')
        
        print(f"✓ Converti: {csv_file} → {excel_file}")
        print(f"   Lignes: {len(df)}")
        print(f"   Colonnes: {len(df.columns)}")
        print(f"   Taille fichier: {os.path.getsize(excel_file)/1024/1024:.2f} MB\n")
        
        return excel_file
        
    except Exception as e:
        print(f"✗ Erreur de conversion pour {csv_file}: {e}")
        return None

def convert_all_csv_in_folder(folder='.'):
    """
    Convertit tous les fichiers CSV d'un dossier en fichiers Excel
    """
    csv_files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
    
    if not csv_files:
        print("✗ Aucun fichier CSV trouvé dans le dossier.")
        return
    
    for csv_file in csv_files:
        csv_path = os.path.join(folder, csv_file)
        excel_file = os.path.join(folder, csv_file.replace('.csv', '.xlsx'))
        convert_csv_to_excel(csv_path, excel_file)

if __name__ == "__main__":
    # Convertir tous les CSV du dossier courant
    convert_all_csv_in_folder()
