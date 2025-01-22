from typing import List, Dict, Any
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, 
    VectorParams,
    PointStruct,
    SearchRequest,
    Filter
)
from pydantic import BaseModel

from news_vector_search.config import settings
from news_vector_search.models.embeddings import EmbeddingService, EmbeddingVector


class SearchResult(BaseModel):
    """Modèle pour les résultats de recherche"""
    title: str
    content: str
    author: str
    published_at: str
    similarity_score: float


class SearchService:
    """Service de recherche d'articles utilisant Qdrant"""
    
    def __init__(
        self,
        collection_name: str = settings.QDRANT_COLLECTION,
        embedding_service: EmbeddingService = None
    ):
        self.collection_name = collection_name
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Initialisation du client Qdrant
        self.client = QdrantClient(
            url=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
        )
        
    def ensure_collection_exists(self):
        """S'assure que la collection existe, la crée si nécessaire"""
        collections = self.client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            print(f"Collection {self.collection_name} créée")
    
    def upsert_articles(self, articles: List[EmbeddingVector], batch_size: int = 100):
        """Insère ou met à jour des articles dans Qdrant"""
        self.ensure_collection_exists()
        
        # Conversion en points Qdrant
        points = [
            PointStruct(
                id=hash(article.article_id),  # Hash stable pour l'ID
                vector=article.vector,
                payload=article.metadata
            )
            for article in articles
        ]
        
        # Insertion par lots
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True
            )
            
        print(f"{len(points)} articles indexés dans Qdrant")
    
    def search(
        self, 
        query: str, 
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[SearchResult]:
        """Recherche les articles les plus pertinents"""
        # Génération de l'embedding de la requête
        query_vector = self.embedding_service.get_embedding(query)
        
        # Recherche dans Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold
        )
        
        # Conversion des résultats
        results = []
        for hit in search_results:
            result = SearchResult(
                title=hit.payload.get("title", ""),
                content=hit.payload.get("content", ""),
                author=hit.payload.get("author", ""),
                published_at=hit.payload.get("published_at", ""),
                similarity_score=hit.score
            )
            results.append(result)
            
        return results
    
    def search_by_date_range(
        self,
        query: str,
        start_date: str,
        end_date: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """Recherche avec filtre de date"""
        query_vector = self.embedding_service.get_embedding(query)
        
        date_filter = Filter(
            must=[
                {
                    "key": "published_at",
                    "range": {
                        "gte": start_date,
                        "lte": end_date
                    }
                }
            ]
        )
        
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=date_filter,
            limit=limit
        )
        
        return [
            SearchResult(
                title=hit.payload.get("title", ""),
                content=hit.payload.get("content", ""),
                author=hit.payload.get("author", ""),
                published_at=hit.payload.get("published_at", ""),
                similarity_score=hit.score
            )
            for hit in search_results
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Récupère des statistiques sur la collection"""
        collection_info = self.client.get_collection(self.collection_name)
        
        return {
            "vectors_count": collection_info.vectors_count,
            "indexed_vectors_count": collection_info.indexed_vectors_count,
            "status": collection_info.status
        }


def main():
    """Test du service de recherche"""
    search_service = SearchService()
    
    # Test de recherche simple
    results = search_service.search("Intelligence artificielle")
    
    print("\nRésultats de recherche:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.title}")
        print(f"Score: {result.similarity_score:.3f}")
        print(f"Auteur: {result.author}")
        print(f"Date: {result.published_at}")
    
    # Affichage des statistiques
    stats = search_service.get_statistics()
    print("\nStatistiques de la collection:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()