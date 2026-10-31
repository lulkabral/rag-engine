import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Configura caminhos base
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Permite importação direta dos módulos independentemente do diretório de execução
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

# Carrega as variáveis de ambiente a partir de .env
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

try:
    from document_processor import load_document, chunk_text
    from vector_store import VectorStore
except ImportError:
    from src.document_processor import load_document, chunk_text
    from src.vector_store import VectorStore

class RAGPipeline:
    def __init__(self, model: str = None):
        self.vector_store = VectorStore()
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "A variável de ambiente 'GROQ_API_KEY' não foi encontrada.\n"
                "Por favor, configure sua chave no arquivo .env na raiz do projeto:\n"
                "GROQ_API_KEY=sua_chave_aqui"
            )
            
        # Conecta à API do Groq usando o modelo configurado
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        self.llm_client = Groq(api_key=api_key)

    def ingest_data(self, file_path: str):
        """Pipeline de ingestão: carrega, divide e armazena os dados."""
        print(f"Iniciando ingestão de dados: {file_path}")
        text = load_document(file_path)
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=80)
        self.vector_store.add_chunks(chunks)
        print("Ingestão concluída. Sistema pronto para consultas.")

    def generate_answer(self, user_query: str, n_results: int = 3) -> str:
        """Recupera contexto e gera uma resposta fundamentada via LLM."""
        # 1. Retrieval (Recuperação)
        relevant_chunks = self.vector_store.search(user_query, n_results=n_results)
        context = "\n\n".join(relevant_chunks)
        
        # 2. Construção do Prompt Aumentado (Augmented)
        prompt = f"""
        Você é um assistente corporativo útil e direto. 
        Responda à pergunta do usuário baseando-se EXCLUSIVAMENTE no contexto fornecido.
        Se a resposta não estiver no contexto, diga que não possui essa informação.
        
        Contexto:
        {context}
        
        Pergunta: {user_query}
        
        Resposta:"""

        # 3. Generation (Geração) usando LLM via Groq
        response = self.llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.2
        )
        
        return response.choices[0].message.content

if __name__ == "__main__":
    # Resolve o caminho do arquivo de dados procurando na raiz do projeto ou no diretório acima
    data_file = PROJECT_ROOT / "data" / "knowledge_base.md"
    if not data_file.exists():
        fallback_path = Path("../data/knowledge_base.md")
        if fallback_path.exists():
            data_file = fallback_path

    if data_file.exists():
        try:
            pipeline = RAGPipeline()
            pipeline.ingest_data(str(data_file))
            
            pergunta = "Qual a política de reembolso em caso de atraso?"
            print(f"\nPergunta: {pergunta}")
            print("-" * 40)
            resposta = pipeline.generate_answer(pergunta)
            print(f"Resposta da IA:\n{resposta}")
        except ValueError as err:
            print(f"\n[Aviso de Configuração] {err}")
    else:
        print(f"Erro: Crie o arquivo {data_file} com a base de conhecimento antes de rodar.")
