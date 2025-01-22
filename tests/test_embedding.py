import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from news_vector_search.data.processor import DataProcessor
from news_vector_search.models.embedding import EmbeddingService

def main():
    # Initialisation des services
    processor = DataProcessor()
    embedding_service = EmbeddingService()
    
    print("Chargement des articles...")
    articles = processor.process_articles()
    print(f"Nombre d'articles chargés : {len(articles)}")
    
    # Test sur 3 articles
    print("\nTest des embeddings sur 3 articles:")
    for i, article in enumerate(articles[:3], 1):
        print(f"\nArticle {i}:")
        print(f"Titre: {article.title}")
        
        # Création de l'embedding
        embedding = embedding_service.get_embedding(article.title + " " + article.content)
        print(f"Dimension du vecteur: {len(embedding)}")
        
        # Test de similarité avec une requête
        query = "guerre Ukraine"
        query_embedding = embedding_service.get_embedding(query)
        similarity = embedding_service.compute_similarity(query_embedding, embedding)
        print(f"Similarité avec '{query}': {similarity:.3f}")

if __name__ == "__main__":
    main()