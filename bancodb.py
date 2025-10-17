import os 
import time
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv 
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma

PASTA_BASE = 'base'

load_dotenv(find_dotenv())
AIMLAPI_KEY = os.getenv("AIMLAPI_KEY")
PASTA_DB = "db_google"
print(f"DEBUG: Chave API carregada? {'Sim' if AIMLAPI_KEY else 'Não'}")

def criar_banco_de_dados():
   
 documentos = carregar_documentos()
 chunks = dividir_chunks(documentos)
 vetorizar_chunks(chunks)

def carregar_documentos():
 carregador = PyPDFDirectoryLoader(PASTA_BASE, glob='*.pdf')
 documentos = carregador.load()
 return documentos  

def dividir_chunks(documentos):
 separador_documentos = RecursiveCharacterTextSplitter( 
    chunk_size = 1000,
    chunk_overlap = 400,
    length_function = len,
    add_start_index = True

 )
 chunks = separador_documentos.split_documents(documentos)
 return chunks

def vetorizar_chunks(chunks):
    print(f"Iniciando a vetorização de {len(chunks)} chunks usando Sentence Transformers (local)...")
    start_time = time.time()

    try:
        embedding_hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if os.path.exists(PASTA_DB):
            print(f"Removendo banco de dados antigo em '{PASTA_DB}'...")
            shutil.rmtree(PASTA_DB)

        print(f"Criando e persistindo o banco de dados em '{PASTA_DB}...")
        db = Chroma.from_documents(
            chunks,
            embedding=embedding_hf,
            persist_directory=PASTA_DB
        )

        end_time = time.time()
        print(f"Vetorização e criação do DB concluídas em {end_time - start_time:.2f} segundos.")

    except Exception as e:
        print(f"ERRO durante a vetorização ou criação do DB local: {e}")

print(f"DEBUG: Chave API carregada? {'Sim' if AIMLAPI_KEY else 'Não'}")
# ADICIONE esta importação no topo do arquivo, se ainda não estiver lá:
from langchain.chains import RetrievalQA

def iniciar_chat():
    print("\nIniciando o sistema de chat...")

    # 1. Verifica se o banco de dados existe
    if not os.path.exists(PASTA_DB):
        print(f"ERRO: Banco de dados '{PASTA_DB}' não encontrado.")
        print("Execute o script com '--create-db' primeiro.")
        return

    # 2. Carrega o banco de dados vetorial
    print("Carregando base de vetores (usando Sentence Transformers)...")
    try:
        embedding_hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory=PASTA_DB, embedding_function=embedding_hf)
        print("Base de dados carregada com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar banco de dados ou embeddings: {e}")
        return

    # 3. Configura a conexão com a AIMLAPI
    try:
        print("Configurando a conexão com a AIMLAPI...")
        aimlapi_base_url = "https://api.aimlapi.com" # Verifique se este URL está correto
        aimlapi_model = "mistralai/Mistral-7B-Instruct-v0.2" # Verifique o nome do modelo

        llm = ChatOpenAI(
            openai_api_key=AIMLAPI_KEY,
            openai_api_base=aimlapi_base_url,
            model_name=aimlapi_model,
            temperature=0.3
        )
        print(f"✅ Conexão com AIMLAPI (modelo '{aimlapi_model}') estabelecida!")

    except Exception as e:
        print(f"ERRO FATAL ao configurar a AIMLAPI: {e}")
        return

    # 4. Cria o "buscador" e a cadeia de Pergunta-Resposta
    retriever = db.as_retriever(search_kwargs={'k': 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # 5. Inicia o loop de perguntas e respostas
    print("\n🚀 Sistema pronto! Faça sua pergunta ou digite 'sair'.")
    while True:
        query = input("\n> ")
        if query.lower() in ('sair', 'exit', 'quit'):
            break

        try:
            print("\n🧠 Pensando... (Consultando AIMLAPI)")
            start = time.time()
            result = qa_chain.invoke({"query": query})
            end = time.time()

            print("\nResposta:")
            print(result.get("result", "Nenhuma resposta recebida da API."))
            print(f"(Tempo de resposta: {end - start:.2f} segundos)")

        except Exception as e:
            print(f"Ocorreu um erro durante a consulta à AIMLAPI: {e}")
