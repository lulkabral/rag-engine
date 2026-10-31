# Corporate RAG Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Cloud-f55036)](https://groq.com/)
[![HuggingFace](https://img.shields.io/badge/Embeddings-Sentence--Transformers-yellow)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Pipeline de **Retrieval-Augmented Generation (RAG)** construído em Python. O sistema processa documentos corporativos locais em Markdown, gera embeddings vetoriais localmente com Hugging Face (`all-MiniLM-L6-v2`), armazena no **ChromaDB** e utiliza o motor de inferência ultra-rápido da **Groq** (OpenAI GPT-OSS) para responder perguntas estritamente fundamentadas no contexto fornecido.

---

## Arquitetura do Sistema

```
[ Documento .md ] 
       │
       ▼
┌─────────────────────────┐
│ 1. Document Processor   │ -> Carrega o texto e fatia em blocos (chunks)
└──────────┬──────────────┘    com sobreposição (overlap)
           │
           ▼
┌─────────────────────────┐
│ 2. Vector Store         │ -> Gera embeddings locais (all-MiniLM-L6-v2)
│    (ChromaDB)           │    e indexa os blocos de texto
└──────────┬──────────────┘
           │ (Busca por Similaridade Semântica)
           ▼
┌─────────────────────────┐
│ 3. Augmented Prompt     │ -> Combina os chunks mais relevantes
└──────────┬──────────────┘    com a pergunta do usuário
           │
           ▼
┌─────────────────────────┐
│ 4. Groq LLM Generation  │ -> Resposta precisa e veloz via OpenAI GPT-OSS
└─────────────────────────┘
```

---

## Estrutura do Repositório

```text
rag-app/
├── data/
│   └── knowledge_base.md        # Documento corporativo com políticas e diretrizes
├── src/
│   ├── __init__.py
│   ├── document_processor.py    # Carregamento e chunking com sliding window
│   ├── vector_store.py          # Gestão do ChromaDB e embeddings Hugging Face
│   └── rag_pipeline.py          # Orquestrador do fluxo RAG e chamada à Groq API
├── .env.example                 # Exemplo de variáveis de ambiente
├── .gitignore                   # Arquivos ignorados pelo Git
├── requirements.txt             # Dependências do projeto
└── README.md                    # Documentação do projeto
```

---

## Como Executar

### 1. Clonar o repositório
```bash
git clone https://github.com/lulkabral/rag-engine.git
cd rag-engine
```

### 2. Criar e ativar o ambiente virtual
```bash
python3 -m venv venv

# No Linux/macOS:
source venv/bin/activate

# No Windows (PowerShell):
.\venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 4. Configurar as variáveis de ambiente
Copie o arquivo `.env.example` para `.env` e insira sua chave da [Groq Cloud](https://console.groq.com/):

```bash
cp .env.example .env
```

Edite o arquivo `.env`:
```ini
GROQ_API_KEY=gsk_sua_chave_aqui_gerada_no_console_groq
GROQ_MODEL=openai/gpt-oss-120b
```

> **Como obter a chave Groq**: Crie uma conta gratuita em [console.groq.com](https://console.groq.com/), vá na aba **API Keys** e gere uma chave.

### 5. Executar o Pipeline RAG

Você pode rodar diretamente da raiz do projeto:
```bash
python src/rag_pipeline.py
```
Ou dentro da pasta `src/`:
```bash
cd src
python rag_pipeline.py
```

---

## Exemplo de Execução

**Entrada:**
```
Pergunta: Qual a política de reembolso em caso de atraso?
```

**Saída esperada:**
```text
Iniciando ingestão de dados: .../data/knowledge_base.md
[6] blocos indexados no banco vetorial.
Ingestão concluída. Sistema pronto para consultas.

Pergunta: Qual a política de reembolso em caso de atraso?
----------------------------------------
Resposta da IA:
De acordo com o documento corporativo, em caso de atrasos imprevistos superiores a 2 horas durante viagens corporativas:
1. É reembolsado até R$ 80,00 por refeição adicional necessária;
2. Se o retorno for inviabilizado no mesmo dia, o colaborador tem direito a pernoite em hotel de até R$ 350,00 por diária;
3. Transportes alternativos decorrentes do atraso são reembolsados integralmente mediante apresentação de comprovante;
4. É obrigatória a apresentação de nota fiscal eletrônica (NFC-e/NFS-e) ou cupom fiscal;
5. As solicitações devem ser enviadas no portal de despesas em até 5 dias úteis após o retorno.
```

---

## Detalhes Técnicos

### 1. Chunking com Janela Deslizante (*Sliding Window*)
Para evitar a perda de contexto semântico entre frases cortadas arbitrariamente, o fatiamento utiliza um algoritmo com sobreposição (*overlap*):
- **Tamanho do Bloco (`chunk_size`):** ~400 caracteres.
- **Sobreposição (`chunk_overlap`):** ~50 caracteres.
- O ponteiro inicial avança `(chunk_size - chunk_overlap)` a cada iteração, garantindo que o final do bloco anterior esteja presente no início do próximo.

### 2. Embeddings Locais (Zero Custo de API na Vetorização)
- Utiliza a biblioteca `sentence-transformers` com o modelo `all-MiniLM-L6-v2`.
- Os vetores são computados localmente no hardware da sua máquina sem cobranças ou limites de taxa externos.
- O ChromaDB gerencia a indexação e executa a busca por proximidade via distância de cosseno.

### 3. Inferência com Groq LPU™
- Os chunks recuperados da busca semântica são injetados em um prompt protegido contra alucinações ("*Responda EXCLUSIVAMENTE no contexto fornecido*").
- A chamada à API da Groq executa a geração em frações de segundo, tornando a experiência quase instantânea.

---

## Personalização

- **Base de conhecimento:** Substitua ou adicione novas informações no arquivo [data/knowledge_base.md](data/knowledge_base.md).
- **Parâmetros de busca:** Altere `n_results` em `vector_store.search()` para controlar quantos blocos de contexto são injetados no prompt.
- **Modelo:** Altere `GROQ_MODEL` no `.env` para outros modelos suportados na Groq (ex: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`).

---

## Licença
Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.
