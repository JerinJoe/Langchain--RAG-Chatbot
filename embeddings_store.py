#######################IMPORTS####################
from PyPDF2 import PdfReader
from sbert import SBERTEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from sentence_transformers import SentenceTransformer
#from langchain_openai import OpenAIEmbeddings
#from langchain_community.embeddings import OllamaEmbeddings
##################################################
#######################CONFIG#####################
# config = dotenv_values(".env")
pdf_path = "path_to_your_pdf"
##################################################

####################MAIN############################
text = ""
pdf_reader = PdfReader(pdf_path)
for page in pdf_reader.pages:
    text += page.extract_text()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = text_splitter.split_text(text)

embeddings = SBERTEmbeddings()
#embeddings = OpenAIEmbeddings(api_key=config["OPENAI_API_KEY"])
#embedding=OllamaEmbeddings(model="llama3",show_progress=True)

vector_store = FAISS.from_texts(chunks, embedding=embeddings)
vector_store.save_local(f"local_doc_embeddings")
#####################################################