from sentence_transformers import SentenceTransformer
from datasets import load_dataset;
import time

ds = load_dataset("agentlans/high-quality-english-sentences", split="test")

data_count = [10, 100, 1000, 10000]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_data(x):
    return ds["text"][:x]

def get_embedding(text, model: SentenceTransformer = model):
    embedding = model.encode(text)
    return embedding


def get_top_n_similar(query_sentence, vector_store, n=3):
    similiarities = {}
    query_embedding = get_embedding(query_sentence)

    for key, value in vector_store.items():
        similiarities[key] = model.similarity(query_embedding, value)

    sorted_similarity = dict(sorted(similiarities.items(), key=lambda x: x[1].item(), reverse=True))
    top_matches = list(sorted_similarity.items())[:n]
    for sentence, score in top_matches:
        print(f"{sentence} - {score.item():.4f}")


for count in data_count:
    print(f"Starting the first {count} data ...")
    
    sentences = get_data(count)
    print(f"Loaded {len(sentences)} sentences")

    embedding = map(get_embedding, sentences)

    vector_store = dict(zip(sentences, embedding))
    start_time = time.perf_counter()
    get_top_n_similar("The ship is not fit for the canal", vector_store)
    end_time = time.perf_counter()
    print(f"Time to process {count} sentences: {(end_time - start_time):.2f}\n")