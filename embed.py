import os
import pandas as pd
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

#load train data
bernie_df = pd.read_csv("Data/PStance/raw_train_bernie.csv")
bernie_df["Source"] = "raw_train_bernie.csv"

biden_df = pd.read_csv("Data/PStance/raw_train_biden.csv")
biden_df["Source"] = "raw_train_biden.csv"

trump_df = pd.read_csv("Data/PStance/raw_train_trump.csv")
trump_df["Source"] = "raw_train_trump.csv"

# Combine them all
df = pd.concat([bernie_df, biden_df, trump_df], ignore_index=True)

# Create documents with metadata
docs = [
    Document(
        page_content=row["Tweet"],
        metadata={"target": row["Target"], "stance": row["Stance"]}
    )
    for _, row in df.iterrows()
]

# Embed with sentence-transformers MiniLM
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in Chroma
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="Data/PStance/chroma_train_db"
)
