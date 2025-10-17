import argparse
from bancodb import criar_banco_de_dados, iniciar_chat

def main():
    parser = argparse.ArgumentParser(description="Chatbot PDF com Embeddings Locais e AIMLAPI")
    parser.add_argument('--create-db', action='store_true',
                        help='Cria/Recria a base vetorial a partir dos PDFs na pasta ./base')
    parser.add_argument('--chat', action='store_true',
                        help='Inicia o chat interativo.')
    args = parser.parse_args()

    if args.create_db:
        criar_banco_de_dados() 
    elif args.chat:
        iniciar_chat() 
    else:
        print("Nenhuma ação especificada. Use --create-db ou --chat.")
        parser.print_help()

if __name__ == "__main__":
    main()