from typing import List, Dict, Any
import torch
from transformers import AutoModel, AutoTokenizer
from pydantic import BaseModel
from functools import lru_cache
import numpy as np
from news_vector_search.config import settings


class EmbeddingVector(BaseModel):
    """Représentation d'un vecteur d'embedding avec métadonnées"""
    article_id: str
    vector: List[float]
    metadata: Dict[str, Any]


class EmbeddingService:
    """Service de gestion des embeddings"""
    
    def __init__(self, model_name: str = settings.MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Utilisation du device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Mode évaluation
        
    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> List[float]:
        """Génère l'embedding pour un texte donné"""
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Déplacement des inputs sur le device approprié
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Utilisation de la moyenne des tokens pour l'embedding final
            embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
            
        return embeddings[0].cpu().numpy().tolist()
    
    def mean_pooling(self, model_output, attention_mask):
        """Moyenne pondérée des embeddings de tokens"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    def batch_get_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> List[List[float]]:
        """Génère les embeddings pour une liste de textes par lots"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = [
                self.get_embedding(text) for text in batch_texts
            ]
            embeddings.extend(batch_embeddings)
            
        return embeddings

    def create_article_embedding(
        self, 
        article_id: str, 
        title: str, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> EmbeddingVector:
        """Crée un vecteur d'embedding pour un article avec ses métadonnées"""
        # Concaténation du titre et du contenu avec plus de poids sur le titre
        full_text = f"{title} {title} {content}"
        vector = self.get_embedding(full_text)
        
        return EmbeddingVector(
            article_id=article_id,
            vector=vector,
            metadata=metadata
        )

    def compute_similarity(
        self, 
        query_vector: List[float], 
        article_vector: List[float]
    ) -> float:
        """Calcule la similarité cosinus entre deux vecteurs"""
        query = np.array(query_vector)
        article = np.array(article_vector)
        
        similarity = np.dot(query, article) / (
            np.linalg.norm(query) * np.linalg.norm(article)
        )
        return float(similarity)


def main():
    """Test du service d'embeddings"""
    service = EmbeddingService()
    
    # Test simple
    text = "Test d'embedding d'un article sur l'intelligence artificielle"
    embedding = service.get_embedding(text)
    print(f"Dimension du vecteur d'embedding: {len(embedding)}")
    
    # Test de traitement par lots
    texts = [
        "Premier article de test",
        "Deuxième article de test",
        "Troisième article de test"
    ]
    embeddings = service.batch_get_embeddings(texts)
    print(f"Nombre d'embeddings générés: {len(embeddings)}")


if __name__ == "__main__":
    main()