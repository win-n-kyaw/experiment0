import chromadb
from datasets import load_dataset

sentence_dataset = load_dataset("agentlans/high-quality-english-sentences", split="train")

client = chromadb.Client()
coll_1 = client.create_collection(name="coll_1")
coll_1.add(
    ids=[str(x) for x in range(len(sentence_dataset["text"][:20]))],
    documents=sentence_dataset["text"][:20],
    metadatas=[ {"sent_id": x} for x in range(len(sentence_dataset["text"][:20]))]
)

result = coll_1.query(
    query_texts="People are excited to welcome the new year",
    where={"sent_id": { "$lt": 10}},
    n_results=3
)

print(result)