import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from news_vector_search.data.processor import DataProcessor

def main():
    processor = DataProcessor()
    
    # 1. Vérifier les fichiers raw disponibles
    raw_files = list(processor.raw_dir.glob("news_*.csv"))
    print("\nFichiers raw disponibles :")
    for f in raw_files:
        print(f"- {f.name}")

    # 2. Lire les données brutes
    raw_df = processor.read_raw_data()
    print(f"\nNombre d'articles avant nettoyage : {len(raw_df)}")
    
    # 3. Suivre le processus de nettoyage
    df_clean = processor.clean_content(raw_df)
    print(f"Après nettoyage du contenu : {len(df_clean)} articles")
    
    df_dates = processor.standardize_dates(df_clean)
    print(f"Après standardisation des dates : {len(df_dates)} articles")
    
    df_selected = processor.select_columns(df_dates)
    print(f"Après sélection des colonnes : {len(df_selected)} articles")
    
    df_no_duplicates = processor.remove_duplicates(df_selected)
    print(f"Après suppression des doublons : {len(df_no_duplicates)} articles")
    
    # 4. Vérifier le traitement final
    articles = processor.process_articles()
    print(f"\nNombre final d'articles traités : {len(articles)}")

    # 5. Examiner les premiers articles
    for i, article in enumerate(articles[:3], 1):
        print(f"\nArticle {i}:")
        print(f"Titre: {article.title}")
        print(f"Auteur: {article.author}")
        print(f"Date: {article.published_at}")
        print("-" * 50)

if __name__ == "__main__":
    main()