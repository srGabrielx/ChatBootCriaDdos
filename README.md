

-----

## 🤖 Chatbot RAG (LangChain + Embeddings Locais + AIMLAPI)

Este projeto implementa um sistema de Chatbot de Perguntas e Respostas baseado em documentos (PDFs) utilizando a arquitetura RAG (Retrieval-Augmented Generation).

Ele usa a biblioteca **LangChain** para orquestração, **HuggingFace Embeddings** (**Sentence Transformers**) para vetorização local, **ChromaDB** como Vector Store e a **AIMLAPI** para o Large Language Model (LLM).

### 🌟 Funcionalidades Principais

  * **RAG com Documentos Locais:** Carrega PDFs da pasta `./base` e utiliza o conteúdo para gerar respostas informadas.
  * **Embeddings Locais:** Utiliza o modelo `sentence-transformers/all-MiniLM-L6-v2` para criar a base vetorial localmente, sem depender de APIs para embeddings.
  * **Vector Store Persistente:** Armazena os vetores no banco de dados local **ChromaDB** (pasta `./db_google`) para reuso.
  * **Integração com AIMLAPI:** Utiliza a AIMLAPI e o modelo `mistralai/Mistral-7B-Instruct-v0.2` para o processamento da linguagem e geração da resposta.

### 🛠️ Tecnologias Utilizadas

| Componente | Tecnologia | Uso Específico |
| :--- | :--- | :--- |
| **Orquestração** | `LangChain` | Criação da cadeia RAG (`RetrievalQA`). |
| **Embeddings** | `HuggingFaceEmbeddings` (Local) | Modelo `all-MiniLM-L6-v2` para criar vetores de documentos. |
| **Vector Store** | `ChromaDB` | Armazenamento persistente dos vetores de documentos. |
| **LLM** | `ChatOpenAI` (via AIMLAPI) | Geração de respostas baseadas no contexto recuperado (`mistralai/Mistral-7B-Instruct-v0.2`). |
| **Processamento PDF** | `PyPDFDirectoryLoader` | Carregamento dos documentos PDF. |
| **Splitter** | `RecursiveCharacterTextSplitter` | Divisão dos documentos em *chunks* para vetorização. |

### 🚀 Começando

Siga estas instruções para configurar e rodar o projeto em sua máquina local.

#### Pré-requisitos

Certifique-se de ter o Python (3.x) e o `pip` instalados.

#### 1\. Clonar o repositório

```bash
git clone https://github.com/wittcher1/ChatBootCriaDdos.git
cd ChatBootCriaDdos
```

#### 2\. Configurar o Ambiente

Crie um ambiente virtual (recomendado) e instale as dependências:

```bash
# Cria e ativa o ambiente virtual (Exemplo com venv)
python -m venv venv
source venv/bin/activate # Linux/macOS
# venv\Scripts\activate # Windows

# Instala as dependências (Assumindo que você tem um requirements.txt ou irá criá-lo)
# Caso não tenha o requirements.txt, instale as libs usadas nos scripts:
pip install langchain-community langchain-chroma langchain-openai langchain-huggingface pypdf python-dotenv
```

#### 3\. Configurar Chave de API

Crie um arquivo chamado **`.env`** na raiz do projeto para armazenar sua chave da AIMLAPI:

```
AIMLAPI_KEY = "SUA_CHAVE_SECRETA_AIMLAPI"
```

**⚠️ Atenção:** ATENÇÃO Não suba o arquivo `.env` para o GitHub. Certifique-se de que ele está no seu `.gitignore`.

#### 4\. Adicionar Documentos

Coloque todos os seus arquivos PDF na pasta **`./base`**.

#### 5\. Criar o Banco de Dados Vetorial

Execute o script com o argumento `--create-db` para carregar os PDFs, criar os embeddings locais e persistir o banco de dados no diretório `./db_google`.

```bash
python main.py --create-db
```

*(Se um banco de dados existente estiver presente em `./db_google`, ele será removido e recriado.)*

#### 6\. Iniciar o Chat

Após a criação do banco de dados, inicie o chat interativo com o argumento `--chat`:

```bash
python main.py --chat
```

O sistema irá carregar a base vetorial, conectar-se à AIMLAPI e você poderá começar a fazer perguntas sobre o conteúdo dos seus PDFs. Digite `sair` para encerrar.

### 📂 Estrutura do Projeto

```
.
├── base/                   # Pasta para os documentos PDF de entrada
│   └── documento.pdf
├── db_google/              # Pasta persistente do ChromaDB (gerada após --create-db)
├── bancodb.py              # Lógica para carregar, dividir e vetorizar documentos, e iniciar o chat.
├── main.py                 # Ponto de entrada que gerencia a execução (--create-db ou --chat).
├── .env                    # Variáveis de ambiente (chave da AIMLAPI)
└── README.md               # Este arquivo.
```

### 🤝 Contribuições

Sinta-se à vontade para sugerir melhorias, reportar bugs ou contribuir com o código.

1.  Faça um Fork do projeto.
2.  Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`).
3.  Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`).
4.  Faça o Push para a branch (`git push origin feature/AmazingFeature`).
5.  Abra um Pull Request.

### 📄 Licença

Este projeto está sob a licença **MIT**.

