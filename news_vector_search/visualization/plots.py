from typing import List, Dict, Any
import numpy as np
from pydantic import BaseModel
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import pandas as pd

from news_vector_search.models.search import SearchResult
from news_vector_search.config import settings


class VisualizationData(BaseModel):
    """Modèle pour les données de visualisation"""
    title: str
    x: float
    y: float
    z: float
    similarity: float
    author: str
    published_at: str


class VisualizationService:
    """Service de visualisation des résultats de recherche"""
    
    def __init__(self):
        self.tsne = TSNE(
            n_components=3,
            perplexity=5,
            random_state=42,
            init='random',
            learning_rate=200
        )
        
    def prepare_embedding_visualization(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> List[VisualizationData]:
        """Prépare les données pour la visualisation"""
        # Réduction de dimension avec t-SNE
        embeddings_array = np.array(embeddings)
        tsne_results = self.tsne.fit_transform(embeddings_array)
        
        # Création des données de visualisation
        viz_data = []
        for i, meta in enumerate(metadata):
            viz_data.append(
                VisualizationData(
                    title=meta.get('title', ''),
                    x=float(tsne_results[i, 0]),
                    y=float(tsne_results[i, 1]),
                    z=float(tsne_results[i, 2]),
                    similarity=meta.get('similarity', 0.0),
                    author=meta.get('author', ''),
                    published_at=meta.get('published_at', '')
                )
            )
        
        return viz_data

    def create_3d_scatter(
        self,
        viz_data: List[VisualizationData],
        query: str = ""
    ) -> go.Figure:
        """Crée un scatter plot 3D interactif"""
        df = pd.DataFrame([data.model_dump() for data in viz_data])
        
        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            color='similarity',
            hover_name='title',
            hover_data=['author', 'published_at', 'similarity'],
            color_continuous_scale='viridis',
            title=f'Articles similaires à la recherche : "{query}"'
        )
        
        # Personnalisation du graphique
        fig.update_traces(
            marker=dict(size=8),
            hovertemplate="<b>%{hovertext}</b><br>" +
                         "Auteur: %{customdata[0]}<br>" +
                         "Date: %{customdata[1]}<br>" +
                         "Score: %{customdata[2]:.3f}<br>"
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='',
                yaxis_title='',
                zaxis_title='',
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return fig
    
    def create_similarity_heatmap(
        self,
        embeddings: List[List[float]],
        titles: List[str]
    ) -> go.Figure:
        """Crée une heatmap de similarité entre les articles"""
        # Calcul de la matrice de similarité
        embeddings_array = np.array(embeddings)
        similarity_matrix = np.dot(embeddings_array, embeddings_array.T)
        
        # Normalisation
        norms = np.linalg.norm(embeddings_array, axis=1)
        similarity_matrix = similarity_matrix / np.outer(norms, norms)
        
        # Création de la heatmap
        fig = px.imshow(
            similarity_matrix,
            labels=dict(x="Article", y="Article", color="Similarité"),
            x=titles,
            y=titles,
            color_continuous_scale="RdBu",
            title="Matrice de similarité entre les articles"
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            margin=dict(l=0, r=0, b=100, t=30)
        )
        
        return fig

    def create_temporal_view(
        self,
        viz_data: List[VisualizationData]
    ) -> go.Figure:
        """Crée une visualisation temporelle des scores de similarité"""
        df = pd.DataFrame([data.model_dump() for data in viz_data])
        df['published_at'] = pd.to_datetime(df['published_at'])
        
        fig = px.scatter(
            df,
            x='published_at',
            y='similarity',
            color='similarity',
            hover_name='title',
            hover_data=['author'],
            color_continuous_scale='viridis',
            title='Distribution temporelle des scores de similarité'
        )
        
        fig.update_traces(marker=dict(size=10))
        fig.update_layout(
            xaxis_title="Date de publication",
            yaxis_title="Score de similarité",
            hovermode='closest'
        )
        
        return fig


def main():
    """Test du service de visualisation"""
    # Données de test
    embeddings = np.random.randn(10, 384)  # 10 articles avec dim 384
    metadata = [
        {
            'title': f'Article {i}',
            'author': f'Auteur {i}',
            'published_at': f'2024-0{i}-01',
            'similarity': np.random.random()
        }
        for i in range(1, 11)
    ]
    
    viz_service = VisualizationService()
    
    # Test de la visualisation 3D
    viz_data = viz_service.prepare_embedding_visualization(embeddings, metadata)
    fig_3d = viz_service.create_3d_scatter(viz_data, "Test query")
    fig_3d.write_html("test_3d_visualization.html")
    
    # Test de la heatmap
    titles = [f'Article {i}' for i in range(1, 11)]
    fig_heatmap = viz_service.create_similarity_heatmap(embeddings, titles)
    fig_heatmap.write_html("test_heatmap.html")
    
    print("Visualisations de test générées !")


if __name__ == "__main__":
    main()