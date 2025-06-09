import os
import pandas as pd
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

#load val data
bernie_df = pd.read_csv("Data/PStance/raw_val_bernie.csv")
bernie_df["Source"] = "raw_val_bernie.csv"

biden_df = pd.read_csv("Data/PStance/raw_val_biden.csv")
biden_df["Source"] = "raw_val_biden.csv"

trump_df = pd.read_csv("Data/PStance/raw_val_trump.csv")
trump_df["Source"] = "raw_val_trump.csv"

# Combine them all
df = pd.concat([bernie_df, biden_df, trump_df], ignore_index=True)
df['label'] = None

def create_prompt(tweet, target):
    prompt = f"""
    You are working to label tweet text as 'FAVOR', 'AGAINST', or 'NEITHER' as they relate to their target. 
    Based on the provided text and target, return only the label.
    text: {tweet}
    target: {target}
    """
    return prompt