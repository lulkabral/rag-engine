from __future__ import annotations
import chromadb
from chromadb.utils import embedding_functions

class VectorStore:
    def __init__(self, collection_name: str = "corporate_knowledge"):
        # Inicializa o ChromaDB em memória (para uso em produção, especifique um diretório)
        self.client = chromadb.Client()
        
        # Modelo de embeddings leve e rápido de código aberto (Hugging Face)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.embedding_fn
        )

    def add_chunks(self, chunks: list[str]):
        """Gera embeddings localmente e adiciona os blocos ao banco vetorial."""
        if not chunks:
            return
            
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            ids=ids
        )
        print(f"[{len(chunks)}] blocos indexados no banco vetorial.")

    def search(self, query: str, n_results: int = 3) -> list[str]:
        """Busca os blocos mais similares semanticamente à query do usuário."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        # Retorna a lista de textos encontrados
        return results['documents'][0] if results.get('documents') and results['documents'] else []
