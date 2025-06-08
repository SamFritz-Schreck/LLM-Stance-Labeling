import pandas as pd
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document

df = pd.read_csv("Data/raw_train_bernie.csv")

# Combine columns into text
texts = df.apply(lambda row: " ".join([str(v) for v in row.values]), axis=1).tolist()
print(texts[0:10])

# Create Document objects
docs = [Document(page_content=text) for text in texts]

# Set up sentence-transformers MiniLM
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create and persist Chroma vectorstore
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory="./chroma_db")
vectorstore.persist()
