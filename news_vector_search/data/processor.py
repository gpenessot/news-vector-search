from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd
from pydantic import BaseModel
from news_vector_search.config import settings


class NewsArticle(BaseModel):
    """Modèle de données pour un article"""
    author: str
    title: str
    published_at: datetime
    content: str
    
    class Config:
        from_attributes = True


class DataProcessor:
    """Classe pour gérer le traitement des données des articles"""
    
    def __init__(self):
        self.raw_dir = settings.RAW_DATA_DIR
        self.processed_dir = settings.PROCESSED_DATA_DIR
        
    def get_all_raw_files(self) -> List[Path]:
        """Récupère tous les fichiers CSV du dossier raw"""
        return sorted(self.raw_dir.glob("news_*.csv"))
    
    def read_raw_data(self) -> pd.DataFrame:
        """Lit et combine tous les fichiers CSV bruts"""
        csv_files = self.get_all_raw_files()
        if not csv_files:
            raise FileNotFoundError("Aucun fichier raw trouvé")
        
        # Lecture et concaténation de tous les fichiers
        dfs = []
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, index_col=0)
                dfs.append(df)
                print(f"Lu {file_path.name} : {len(df)} articles")
            except Exception as e:
                print(f"Erreur lors de la lecture de {file_path.name}: {e}")
                continue
        
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df

    def clean_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie le contenu des articles"""
        print(f"Articles avant nettoyage: {len(df)}")
        
        # Suppression des articles de Google News
        df = df[df['source'] != 'google-news-fr']
        print(f"Articles après filtrage Google News: {len(df)}")
        
        # Suppression des lignes avec contenu ou auteur manquant
        df = df.dropna(subset=["content", "author"])
        print(f"Articles après suppression des NA: {len(df)}")
        
        # Nettoyage du contenu spécifique
        df = df[~df["content"].str.startswith("Search\r\nDirect", na=False)]
        print(f"Articles après nettoyage contenu: {len(df)}")
        
        return df

    def standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise les dates de publication"""
        df = df.copy()
        df["publishedAt"] = pd.to_datetime(df["publishedAt"]).dt.tz_localize(None)
        return df

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sélectionne et renomme les colonnes pertinentes"""
        columns_mapping = {
            "author": "author",
            "title": "title",
            "publishedAt": "published_at",
            "content": "content"
        }
        return df[list(columns_mapping.keys())].rename(columns=columns_mapping)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les doublons"""
        initial_len = len(df)
        df = df.drop_duplicates(subset=["title", "content"])
        print(f"Doublons supprimés: {initial_len - len(df)}")
        return df

    def process_articles(self) -> List[NewsArticle]:
        """Traite tous les articles et retourne une liste d'objets NewsArticle"""
        print("Début du traitement des articles...")
        
        # Lecture des données
        df = self.read_raw_data()
        print(f"Total articles bruts: {len(df)}")
        
        # Application des transformations
        df = (df
              .pipe(self.clean_content)
              .pipe(self.standardize_dates)
              .pipe(self.select_columns)
              .pipe(self.remove_duplicates)
        )
        
        # Conversion en objets Pydantic
        articles = [
            NewsArticle(**row.to_dict()) 
            for _, row in df.iterrows()
        ]
        
        print(f"Traitement terminé. Articles finaux: {len(articles)}")
        return articles


def main():
    """Point d'entrée pour le test"""
    processor = DataProcessor()
    try:
        articles = processor.process_articles()
        print(f"\nStatistiques finales:")
        print(f"Nombre total d'articles traités: {len(articles)}")
        
        # Affichage des 3 premiers articles
        print("\nExemples d'articles:")
        for i, article in enumerate(articles[:3], 1):
            print(f"\nArticle {i}:")
            print(f"Titre: {article.title}")
            print(f"Auteur: {article.author}")
            print(f"Date: {article.published_at}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")

if __name__ == "__main__":
    main()