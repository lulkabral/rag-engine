from __future__ import annotations
import os

def load_document(file_path: str) -> str:
    """Lê um arquivo de texto e retorna o seu conteúdo completo."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Divide um texto grande em blocos menores com uma área de sobreposição (overlap).
    
    Args:
        text: Texto completo a ser fatiado.
        chunk_size: Quantidade de caracteres em cada bloco.
        chunk_overlap: Quantidade de caracteres que se sobrepõem entre blocos adjacentes.
        
    Returns:
        Lista de strings contendo os chunks gerados.
    """
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size deve ser maior que chunk_overlap.")
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Avança o tamanho do chunk, recuando a zona de sobreposição
        start += (chunk_size - chunk_overlap)
        
    return chunks
