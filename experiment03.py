import chromadb
from chromadb.utils.batch_utils import create_batches
from datasets import load_dataset;
import time

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="sentences")

def get_data(x):
    return ds["text"][:x]
ds = load_dataset("agentlans/high-quality-english-sentences", split="test")
data_count = [10, 100, 1000, 10000]

for count in data_count:
    id = [str(x) for x in range(count)]
    data = get_data(count)

    if count > 5461:
        batches = create_batches(api=chroma_client, ids=id, documents=data)
        for batch in batches:
            collection.add(
                ids=batch[0],
                documents=batch[3]
            )
    else:
        collection.add(ids=id, documents=data)
    print(f"Number of Rows - {collection.count()}")
    start_time = time.perf_counter()
    results = collection.query(
        query_texts=["The ship is not fit for the canal"],
        n_results=3
    )
    end_time = time.perf_counter()
    print(f"Time Taken: {(end_time - start_time):.2f}")
    print(results)