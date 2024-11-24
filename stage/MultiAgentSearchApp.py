import streamlit as st 
from transformers import pipeline
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pdfplumber
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import YouTubeSearchTool
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from youtubesearchpython import VideosSearch

wikipedia = WikipediaAPIWrapper()

# API keys and global configurations
GOOGLE_API_KEY = 'AIzaSyAD35w9sxvo7DTnL85e0F5BsBsTH60g8xY'

def main():
    st.title("Multi-Agent Search Application")
    st.sidebar.title("Navigation")
    
    # Navigation options
    options = [
        "Accueil",
        "Recherche dans un fichier .txt",
        "Recherche dans un fichier PDF",
        "Recherche dans une URL",
        "Recherche DuckDuckGo",
        "Recherche Google",
        "Requête Wikipédia",
        "Recherche YouTube"
    ]
    choice = st.sidebar.radio("Choisissez une section", options)

    if choice == "Accueil":
        accueil()
    elif choice == "Recherche dans un fichier .txt":
        recherche_txt()
    elif choice == "Recherche dans un fichier PDF":
        recherche_pdf()
    elif choice == "Recherche dans une URL":
        recherche_url()
    elif choice == "Recherche DuckDuckGo":
        recherche_duckduckgo()
    elif choice == "Recherche Google":
        recherche_google()
    elif choice == "Requête Wikipédia":
        recherche_wikipedia()
    elif choice == "Recherche YouTube":
        recherche_youtube()

# Pages principales
def accueil():
    st.write("""
    ### Bienvenue dans l'application MultiAgentSearchApp !
    Cette application regroupe plusieurs fonctionnalités :
    - Recherche dans des fichiers texte ou PDF.
    - Recherche en ligne via DuckDuckGo, Google, Wikipédia, ou YouTube.
    - Accès à des outils avancés pour extraire des informations précises.

    Utilisez la barre de navigation pour commencer.
    """)

def recherche_txt():
    st.header("Recherche dans un fichier .txt")
    uploaded_file = st.file_uploader("Chargez un fichier .txt", type="txt")

    if uploaded_file is not None:
        loader = TextLoader(uploaded_file)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=GOOGLE_API_KEY
        )

        db = FAISS.from_documents(docs, embeddings)
        st.success("Indexation terminée.")

        question = st.text_input("Posez votre question :")
        if question:
            results = db.similarity_search(question, k=5)
            context = " ".join([doc.page_content for doc in results])
            qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            answer = qa_pipeline(question=question, context=context)
            st.write(f"**Réponse :** {answer['answer']}")

def recherche_pdf():
    st.header("Recherche dans un fichier PDF")
    uploaded_file = st.file_uploader("Chargez un fichier PDF", type="pdf")

    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            text = "".join([page.extract_text() for page in pdf.pages])

        st.text_area("Contenu extrait", text[:500])  # Affiche un extrait

        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=GOOGLE_API_KEY
        )

        db = FAISS.from_documents(chunks, embeddings)
        st.success("Indexation terminée.")

        question = st.text_input("Posez votre question :")
        if question:
            results = db.similarity_search(question, k=5)
            context = " ".join([chunk.page_content for chunk in results])
            qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            answer = qa_pipeline(question=question, context=context)
            st.write(f"**Réponse :** {answer['answer']}")

def recherche_url():
    st.header("Recherche dans une URL")
    url = st.text_input("Entrez l'URL :")

    if url:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        text = " ".join(paragraphs)

        st.text_area("Contenu extrait", text[:500])

def recherche_duckduckgo():
    st.header("Recherche DuckDuckGo")
    query = st.text_input("Entrez votre requête :")

    if query:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        st.write("**Résultats :**")
        st.write(results)

def recherche_google():
    st.header("Recherche Google")
    query = st.text_input("Entrez votre requête :")

    if query:
        results = [result for result in search(query, num_results=10)]
        st.write("**Résultats :**")
        for result in results:
            st.write(result)

def recherche_wikipedia():
    st.header("Requête Wikipédia")
    query = st.text_input("Entrez votre requête :")

    if query:
        wikipedia = WikipediaAPIWrapper()
        result = wikipedia.run(query)
        st.write("**Résultat :**")
        st.write(result)

def recherche_youtube():
    st.header("Recherche YouTube")
    query = st.text_input("Entrez votre requête :")

    if query:
        # Effectuer la recherche
        search = VideosSearch(query, limit = 5)
        results = search.result()  # Méthode correcte pour obtenir les résultats

        st.write("**Résultats :**")
        for result in results['result']:
            st.write(f"Title: {result['title']}, URL: {result['link']}")

if __name__ == "__main__":
    main()
