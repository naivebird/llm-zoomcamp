import numpy as np
import requests
import pandas as pd
import minsearch
from qdrant_client import QdrantClient, models
from minsearch import VectorSearch
from qdrant_client.fastembed_common import QueryResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'
docs_url = url_prefix + 'search_evaluation/documents-with-ids.json'
documents = requests.get(docs_url).json()

ground_truth_url = url_prefix + 'search_evaluation/ground-truth-data.csv'
df_ground_truth = pd.read_csv(ground_truth_url)
ground_truth = df_ground_truth.to_dict(orient='records')

from tqdm.auto import tqdm

def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)

def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }


def qdrant_evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d.payload['id'] == doc_id for d in results.points]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }

def minsearch_search(question, course, index):
    boost = {
        'question': 1.5,
        'section': 0.1
    }
    results = index.search(
        query=question,
        filter_dict={'course': course},
        boost_dict=boost,
        num_results=5
    )
    return results

def minsearch_vector_search(question, course, index):
    results = index.search(
        query_vector=question,
        filter_dict={'course': course},
        num_results=5
    )
    return results

def index_documents(documents):
    index = minsearch.Index(text_fields=['question', 'text', 'section'],
                            keyword_fields=['course', 'id'])
    index.fit(documents)
    return index

def index_vector_documents(documents):
    texts = []

    for doc in documents:
        t = doc['question'] + ' ' + doc['text']
        texts.append(t)

    pipeline = make_pipeline(
        TfidfVectorizer(min_df=3),
        TruncatedSVD(n_components=128, random_state=1)
    )
    X = pipeline.fit_transform(texts)

    vindex = VectorSearch(keyword_fields={'course'})
    vindex.fit(X, documents)
    return vindex, pipeline

def get_qdrant_client():
    return QdrantClient("http://localhost:6333")


def create_qdrant_collection(model_handle, collection_name, documents):
    client = get_qdrant_client()
    EMBEDDING_DIMENSIONALITY = 512

    client.delete_collection("zoomcamp-rag")  # Clean up if the collection already exists

    # Create the collection with specified vector parameters
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSIONALITY,  # Dimensionality of the vectors
            distance=models.Distance.COSINE  # Distance metric for similarity search
        )
    )

    points = []
    id = 0
    for doc in documents:
        point = models.PointStruct(
            id=id,  # Use the document ID as the point ID
            vector=models.Document(text=doc['question'] + ' ' + doc['text'], model=model_handle),
            # embed text locally with "jinaai/jina-embeddings-v2-small-en" from FastEmbed
            payload={
                "text": doc['text'],
                "question": doc['question'],
                "id": doc['id'],
                "course": doc['course'],
            }  # save all needed metadata fields
        )
        points.append(point)
        id += 1

    client.upsert(
        collection_name=collection_name,
        points=points
    )


def qdrant_search(query, course, collection_name, model_handle, limit=5):
    client = get_qdrant_client()
    results = client.query_points(
        collection_name=collection_name,
        query=models.Document(  # embed the query text locally with "jinaai/jina-embeddings-v2-small-en"
            text=query,
            model=model_handle
        ),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key='course',
                    match=models.MatchValue(value=course)
                )
            ]
        ),
        limit=limit,  # top closest matches
        with_payload=True  # to get metadata in the results
    )
    return results

def cosine(u, v):
    u_norm = np.sqrt(u.dot(u))
    v_norm = np.sqrt(v.dot(v))
    return u.dot(v) / (u_norm * v_norm)


if __name__ == '__main__':
    # index = index_documents(documents)
    # results = evaluate(ground_truth=ground_truth, search_function=lambda q: minsearch_search(q['question'], q['course'], index))
    # print(results)

    # vindex, pipeline = index_vector_documents(documents)
    #
    # results = evaluate(ground_truth=ground_truth, search_function=lambda q: minsearch_vector_search(pipeline.transform([q['question']]), q['course'], vindex))
    # print(results)

    # collection_name = "zoomcamp-rag"
    # model_handle = "jinaai/jina-embeddings-v2-small-en"
    #
    # create_qdrant_collection(model_handle=model_handle,
    #                          collection_name=collection_name,
    #                          documents=documents)
    #
    # results = qdrant_evaluate(ground_truth=ground_truth,
    #                           search_function=lambda q: qdrant_search(q['question'], q['course'], collection_name, model_handle))
    # print(results)

    results_url = url_prefix + 'rag_evaluation/data/results-gpt4o-mini.csv'
    df_results = pd.read_csv(results_url)

    # pipeline = make_pipeline(
    #     TfidfVectorizer(min_df=3),
    #     TruncatedSVD(n_components=128, random_state=1)
    # )
    #
    # pipeline.fit(df_results.answer_llm + ' ' + df_results.answer_orig + ' ' + df_results.question)
    #
    # v_llm = pipeline.transform(df_results.answer_llm)
    # v_orig = pipeline.transform(df_results.answer_orig)
    # cosine = np.array([cosine(u, v) for u, v in zip(v_llm, v_orig)])
    # avg_cosine = np.mean(cosine)
    # print(f"Average cosine similarity: {avg_cosine:.4f}")

    from rouge import Rouge

    rouge_scorer = Rouge()

    scores = rouge_scorer.get_scores(df_results.answer_llm, df_results.answer_orig)
    rouge_1_f1 = np.mean([score['rouge-1']['f'] for score in scores])
    print('ROUGE-1 F1: ', rouge_1_f1)

