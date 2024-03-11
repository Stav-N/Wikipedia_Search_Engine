# from utils import *
from inverted_index_gcp_project import InvertedIndex, MultiFileReader
# from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import numpy as np
import pandas as pd
from collections import Counter
import pickle
from contextlib import closing
from google.cloud import storage
import zipfile



def load_pkl_from_bucket(path, bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(path)
    contents = blob.download_as_bytes()
    return pickle.loads(contents)

# GLOBAL VARIABLES
BLOCK_SIZE = 1999998
TUPLE_SIZE = 6       
TF_MASK = 2 ** 16 - 1
bucket_name = 'ir-project-bucket-2024'
doc_id_title_dict = load_pkl_from_bucket('id2title.pkl', 'ir-project-bucket-2024')
doc_id_pagerank_dict = load_pkl_from_bucket('id2pagerank.pkl', 'ir-project-bucket-2024')
title_index = load_pkl_from_bucket('title_postings/index.pkl', 'ir-project-bucket-2024')
body_index = load_pkl_from_bucket('body_postings/index.pkl', 'ir-project-bucket-2024')

corpus = ["category", "references", "also", "external", "links",
                      "may", "first", "see", "history", "people", "one", "two",
                      "part", "thumb", "including", "second", "following",
                      "many", "however", "would", "became"]
english_stop_words = frozenset(stopwords.words('english'))
# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()


def search_backend(query):
    tokenized_query = tokenize(query)
    if len(tokenized_query) <= 3:
        return combined_similarity(query)
    else:
        return search_body_backend(query, retrival_model='simple_tf')


def id2title(top_n_docs):
    return [(str(doc_id), doc_id_title_dict[doc_id]) for doc_id, score in top_n_docs]

def sort_by_score(docs_scores_dict):
    return sorted([(doc_id, score) for doc_id, score in docs_scores_dict.items()], key=lambda x: x[1], reverse=True)

def combined_similarity(query, retrive_n_docs=30):
    #title_bm25 = search_title_backend(query, retrival_model='bm25')
    top_n_docs, title_bm25 = search_title_backend_pagerank(query, retrival_model='bm25')
    combined_scores = {}
    for doc_id in title_bm25.keys():
        if doc_id in doc_id_pagerank_dict:
            bm25_score = title_bm25[doc_id]
            page_rank_score = doc_id_pagerank_dict[doc_id]
            combined_score = 0.7 * bm25_score + 0.3 * page_rank_score  # Adjust weights as needed
            #combined_score = 0.6 * bm25_score + 0.4 * page_rank_score
            combined_scores[doc_id] = combined_score
    top_n_docs = sort_by_score(combined_scores)[:retrive_n_docs]
    return id2title(top_n_docs)

def search_title_backend(query, retrival_model='cossimilarity'):
    tokenized_query = tokenize(query)
    query_tokens = list(filter(lambda x: x in title_index.df, tokenized_query))
    unique_tokens = set(query_tokens)
    if len(unique_tokens) == 0:
        return []
    
    if retrival_model == 'cossimilarity':
        doc_token_tfidf, tfidf_query, unique_docs = calculate_tfidf(query_tokens, unique_tokens, title_index)
        top_n_docs = cosine_similarity(doc_token_tfidf, tfidf_query, unique_docs, unique_tokens)
    
    if retrival_model == 'simple_tf':
        top_n_docs = simple_tf(unique_tokens, title_index)

    if retrival_model == 'bm25':
        doc_token_tf, tf_query, unique_docs = calculate_tf(query_tokens, unique_tokens)
        # top_n_docs = bm25_similarity(doc_token_tf, tf_query, unique_docs, unique_tokens, title_index, k1=1.5, k3=1.0, b=0.75, retrive_n_docs=30)
        top_n_docs, scores = bm25_similarity(doc_token_tf, tf_query, unique_docs, unique_tokens, title_index, k1=1.5, k3=1.0, b=0.75, retrive_n_docs=30)

    return id2title(top_n_docs)

def search_title_backend_pagerank(query, retrival_model='bm25'):
    tokenized_query = tokenize(query)
    query_tokens = list(filter(lambda x: x in title_index.df, tokenized_query))
    unique_tokens = set(query_tokens)
    if len(unique_tokens) == 0:
        return []
    
    if retrival_model == 'bm25':
        doc_token_tf, tf_query, unique_docs = calculate_tf(query_tokens, unique_tokens)
        # top_n_docs = bm25_similarity(doc_token_tf, tf_query, unique_docs, unique_tokens, title_index, k1=1.5, k3=1.0, b=0.75, retrive_n_docs=30)
        top_n_docs, scores = bm25_similarity(doc_token_tf, tf_query, unique_docs, unique_tokens, title_index, k1=1.5, k3=1.0, b=0.75, retrive_n_docs=30)

    return id2title(top_n_docs), scores

def search_body_backend(query, retrival_model='cossimilarity'):
    tokenized_query = tokenize(query)
    query_tokens = list(filter(lambda x: x in body_index.df, tokenized_query))
    #query_tokens = list(filter(lambda x: x in title_index.df, tokenized_query))
    unique_tokens = set(query_tokens)
    if len(unique_tokens) == 0:
        return []
    
    if retrival_model == 'cossimilarity':
        doc_token_tfidf, tfidf_query, unique_docs = calculate_tfidf(query_tokens, unique_tokens, body_index)
        top_n_docs = cosine_similarity(doc_token_tfidf, tfidf_query, unique_docs, unique_tokens)
    
    if retrival_model == 'simple_tf':
        top_n_docs = simple_tf(unique_tokens, body_index)
    
    return id2title(top_n_docs)


def simple_tf(unique_tokens, index, retrive_n_docs=30):
    
    docs_scores_dict = {}
    for token in unique_tokens:
        if token in index.df:
            _, curr_unique_docs = get_doc_token_tfidf(token, index)
            for doc_id in curr_unique_docs:
                docs_scores_dict[doc_id] = docs_scores_dict.get(doc_id, 0) + 1
    top_n_docs = sort_by_score(docs_scores_dict)[:retrive_n_docs]
    
    return top_n_docs


def calculate_average_doc_length(index):
    # Calculate the total length of all documents
    total_length = sum(index.doc_len.values())
    
    # Calculate the number of documents
    num_docs = len(index.doc_len)
    
    # Calculate and return the average length
    average_doc_len =  total_length / num_docs if num_docs else 0

    return average_doc_len


def bm25_similarity(doc_token_tf, query_tf, unique_docs, unique_tokens, index, k1=1.5, k3=1.0, b=0.75, retrive_n_docs=30):
    #doc_token_tf {(doc_id,token) : tf}
    #query_tf {token: tf}
    average_doc_len = calculate_average_doc_length(index)
    # Compute document length normalization factor B
    B = 1 - b + b * (len(doc_token_tf) / (average_doc_len if average_doc_len > 0 else 1))

    # Compute term frequency (TF) and inverse document frequency (IDF) for query tokens
    token_counts = Counter(unique_tokens)
    token_vector = np.array(sorted(set(unique_tokens)))
    tf = np.array([token_counts[token] / len(unique_tokens) for token in token_vector])

    # Compute BM25 score for each document
    scores = {}
    for token in unique_tokens:
        if token in index.df:
            df = index.df[token] #df = Number of documents containing token t
            idf = np.log10(len(index.doc_len) / df)
            _, curr_unique_docs = get_doc_token_tf(token)
            for doc_id in curr_unique_docs:
            # for doc_id in unique_docs:
                key = (doc_id, token)
                if key in doc_token_tf:
                    score = (((k1 +1)*doc_token_tf[(doc_id, token)]) / B * k1 + doc_token_tf[(doc_id, token)]) * idf * (((k3 + 1) * query_tf[token]) / (k3 + query_tf[token]))
                    
                    scores[doc_id] = scores.get(doc_id, 0) + score
    top_n_docs = sort_by_score(scores)[:retrive_n_docs]

    return top_n_docs, scores



def cosine_similarity(doc_token_tfidf, query_tfidf, unique_docs, unique_tokens, retrive_n_docs=30):
    # Pre-calculate the norm of the query TF-IDF vector for normalization
    query_norm = sum(value ** 2 for value in query_tfidf.values()) ** 0.5
    
    docs_scores_dict = {}
    for doc_id in unique_docs:
        curr_docs_scores_dict = 0
        doc_norm = 0  # For calculating the norm of the document TF-IDF vector
        for token in unique_tokens:
            key = (doc_id, token)
            if key in doc_token_tfidf:
                curr_docs_scores_dict += query_tfidf[token] * doc_token_tfidf[key]
                doc_norm += doc_token_tfidf[key] ** 2
        doc_norm = doc_norm ** 0.5
        # Normalize the similarity score
        if doc_norm > 0:  # Avoid division by zero
            docs_scores_dict[doc_id] = curr_docs_scores_dict / (query_norm * doc_norm)
        else:
            docs_scores_dict[doc_id] = 0
    top_n_docs = sort_by_score(docs_scores_dict)[:retrive_n_docs]
    return top_n_docs



def calculate_tfidf(query_tokens, unique_tokens, index):
    doc_token_tfidf, query_tfidf, unique_docs = {}, {}, set()
    
    # Calculate documents tf-idf
    for token in unique_tokens:
        curr_doc_token_tfidf, curr_unique_docs = get_doc_token_tfidf(token, index)
        doc_token_tfidf.update(curr_doc_token_tfidf)
        unique_docs = unique_docs.union(curr_unique_docs)
    
    # Calculate query tf-idf
    token_counts = Counter(query_tokens)
    token_vector = np.array(sorted(set(query_tokens)))
    tf = np.array([token_counts[token] / len(query_tokens) for token in token_vector])
    df = np.array([index.df[token] for token in token_vector])
    idf = np.log10(len(index.doc_len) / df)
    query_tfidf_scores = tf * idf
    query_tfidf = dict(zip(token_vector, query_tfidf_scores))
    
    return doc_token_tfidf, query_tfidf, unique_docs
    

def calculate_tf(query_tokens, unique_tokens):
    doc_token_tf, query_tf, unique_docs = {}, {}, set()
    
    # Calculate documents tf-idf
    for token in unique_tokens:
        curr_doc_token_tf, curr_unique_docs = get_doc_token_tf(token)
        doc_token_tf.update(curr_doc_token_tf)
        unique_docs = unique_docs.union(curr_unique_docs)
    
    # Calculate query tf-idf
    token_counts = Counter(query_tokens)
    token_vector = np.array(sorted(set(query_tokens)))
    tf = np.array([token_counts[token] / len(query_tokens) for token in token_vector])
    query_tf_scores = tf
    query_tf = dict(zip(token_vector, query_tf_scores))
    
    return doc_token_tf, query_tf, unique_docs

def get_doc_token_tf(token):
    with closing(MultiFileReader(base_dir='postings_gcp/', bucket_name='ir_project_title_tfidf')) as reader:
        locs = title_index.posting_locs[token]
        b = reader.read(locs, title_index.df[token] * TUPLE_SIZE)
        curr_doc_token_tfidf = {}
        curr_unique_docs = []
        for i in range(title_index.df[token]):
            doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
            curr_unique_docs.append(doc_id)
            
            tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
            try:
                curr_doc_token_tfidf[(doc_id, token)] = (tf / title_index.doc_len[doc_id])
            except KeyError:
                curr_doc_token_tfidf[(doc_id, token)] = (tf / 5)
    return curr_doc_token_tfidf, curr_unique_docs

    
def get_doc_token_tfidf(token, index):
    idf = np.log10(len(index.doc_len)/np.array(index.df[token]))
    with closing(MultiFileReader(base_dir='postings_gcp/', bucket_name='ir_project_title_tfidf')) as reader:
    #with closing(MultiFileReader(base_dir='postings_gcp/', bucket_name='ir_project_body_tfidf')) as reader:
        locs = index.posting_locs[token]
        b = reader.read(locs, index.df[token] * TUPLE_SIZE)
        curr_doc_token_tfidf = {}
        curr_unique_docs = []
        for i in range(index.df[token]):
            doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
            curr_unique_docs.append(doc_id)
            
            tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
            try:
                curr_doc_token_tfidf[(doc_id, token)] = (tf / index.doc_len[doc_id]) * idf
            except KeyError:
                curr_doc_token_tfidf[(doc_id, token)] = (tf / 5) * idf
    return curr_doc_token_tfidf, curr_unique_docs


def get_stop_words(english_stopwords = False, corpus_stopwords = False):
    stop_words = frozenset([])
    if english_stopwords:
        stop_words = stop_words.union(english_stop_words)
    if corpus_stopwords:
        stop_words = stop_words.union(corpus)
    return stop_words


def tokenize(text):
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    filtered_tokens = [token for token in tokens if token not in get_stop_words(True, True)]
    return filtered_tokens

