""" Script to run wiki search offline
"""
# Import necessary libraries
from datasets import load_dataset
import pandas as pd
import os
import numpy as np
from config import WIKI_DATA, EMB_MODEL, MAX_DOCS, TOP_K, EMB_PATH
from sentence_transformers import SentenceTransformer, util

def load_data_complete(max_docs: int, wiki_emb: str) -> dict:
    """ Load the complete wikipedia data.

    Args:
        max_docs (int): maximum documents to load.
        wiki_emb (str): name of the dataset.

    Returns:
        dict: dict of documents.
    """
    # Load max documents
    docs_stream = load_dataset(wiki_emb, 'en', split="train")
    
    # Store the documents as dataframe
    docs = []

    for doc in docs_stream:
        docs.append(doc)
        if len(docs) >= max_docs:
            break
    del docs_stream
    return docs

def get_embeddings(emb_model:str, 
                   docs: dict, 
                   query_str: bool == False) -> np.array:
    """ Get the embeddings.

    Args:
        emb_model (str): name of the model.
        docs (dict): dict of documents.
        query_str (bool, optional): whether it's a query or text. 
                                    Defaults to =False.

    Returns:
        np.array: embeddings of the data.
    """
    # Get the model
    model = SentenceTransformer(emb_model) 
    instruction = "Answer the following question correctly. "

    # Get the embeddings of the query
    if query_str:
        embeddings = model.encode([instruction+q for q in docs], 
                                  normalize_embeddings=True,
                                  convert_to_tensor=True)
    else:
        # Get the embeddings of the data
        texts = []
        for doc in docs:
            texts.append(doc['text'])
        # Create embeddings if it doesn't exist
        if not os.path.exists(EMB_PATH):
            embeddings= model.encode(texts, normalize_embeddings=True,
                                    convert_to_tensor=True)
            np.save(EMB_PATH, embeddings)
        else:
            # Load the embeddings if it exists
            embeddings = np.load(EMB_PATH)
    return embeddings


def get_top_k(query: str, docs: dict, q_emb, d_emb, top_k: int) -> (list, list):
    """ Get top k documents based on similarity score

    Args:
        query (str): user query.
        docs (dict): document dictionary.
        q_emb (_type_): query embedding.
        d_emb (_type_): document embedding
        top_k (int): top k results.
    
    Returns:
        (list, list): titles, texts.
    """
    # Compute dot score between query embedding and document embeddings
    # dot_scores = q_emb @ d_emb.T
    # top_k = torch.topk(torch.tensor(dot_scores), k=top_k)
    top_k = util.semantic_search(q_emb, d_emb, top_k=top_k)
    hits = top_k[0]

    # Print results
    titles = []
    texts = []
    indices = []
    for doc in hits:
        indices.append(doc['corpus_id'])

    print("Query:", query)
    for doc_id in indices:
        titles.append(docs[doc_id]['title'])
        texts.append(docs[doc_id]['text'])
        print(docs[doc_id]['title'])
        print(docs[doc_id]['text'], "\n")
    return titles, texts

def main(query: str):
    """ Main method to call the workflow

    Args:
        query (str): user query.
    """
    # Get query embeddings
    query_embedding = get_embeddings(EMB_MODEL, [query], True)

    # Load data
    docs = load_data_complete(max_docs=MAX_DOCS, wiki_emb=WIKI_DATA)

    # Get docs embeddings
    doc_embeddings = get_embeddings(emb_model=EMB_MODEL, docs=docs, query_str=False)

    # Get top k documents matching the query
    title, text = get_top_k(query, docs, query_embedding, doc_embeddings, TOP_K)
    print(title, text)

