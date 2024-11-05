from sentence_transformers import SentenceTransformer

class SBERTEmbeddings:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts)
    
    def embed_query(self, query):
        return self.model.encode([query])[0]

    def __call__(self, text):
        return self.embed_query(text)