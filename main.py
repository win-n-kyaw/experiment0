from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import time
import chromadb
from chromadb.utils.batch_utils import create_batches

sentence_dataset = load_dataset("agentlans/high-quality-english-sentences", split="train")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="sentences")

def initialize_vector_db(sentences, embeddings):
    ids = [ str(x) for x in range(len(sentences))]
    vector_db = {
        'ids': ids,
        'sentences': sentences,
        'embeddings': embeddings
    }
    return vector_db

def query(db, query_embedding, model, top_n=3):
    similarities = []
    for i, emb in enumerate(db['embeddings']):
        sim = model.similarity(query_embedding, emb)
        similarities.append((i, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    results = []
    for i, sim in similarities[:top_n]:
        results.append({
            'id': db['ids'][i],
            'sentence': db['sentences'][i],
            'similarity': float(sim)
        })
    return results

def add_collection(sentences, collection , client):
    ids = [str(x) for x in range(len(sentences))]
    if len(sentences) > 5000:
        batches = create_batches(api=client, ids=ids, documents=sentences)
        for batch in batches:
            collection.add(
                ids=batch[0],
                documents=batch[3]
            )
        return collection
    else:
        collection.add(ids=ids, documents=sentences)
        return collection
    

def main():
    dataset_options = [10, 100, 1000, 10000]
    choice = 10000
    query_sentence = ["This is not the product I asked for"]
    sentences = sentence_dataset["text"][:choice]
    
    # Naive
    naive_start_time = time.perf_counter()

    embedding = embedding_model.encode(sentences)
    vector_db = initialize_vector_db(sentences, embedding)
    naive_init_time = time.perf_counter()

    query_embedding = embedding_model.encode(query_sentence)
    naive_results = query(vector_db, query_embedding, embedding_model, 5)
    naive_result_time = time.perf_counter()

    print(naive_results)

    # #Chromadb
    chroma_start_time = time.perf_counter()

    chroma_collection = add_collection(sentences, collection, chroma_client)
    chroma_init_time = time.perf_counter()

    chroma_results = chroma_collection.query(
        query_texts=query_sentence,
        n_results=5
    )
    chroma_result_time = time.perf_counter()
    print(chroma_results)

    print(f"Summary \n Naive: init-{(naive_init_time - naive_start_time):.4f}s query-{(naive_result_time - naive_init_time):.4f}")
    print(f"\n Chroma: init-{(chroma_init_time - chroma_start_time):.4f}s query-{(chroma_result_time - chroma_init_time):.4f}")

if __name__ == "__main__":
    main()