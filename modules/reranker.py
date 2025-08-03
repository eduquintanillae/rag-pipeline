from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model):
        self.model = model
        if self.model == "cross-encoder/ms-marco-MiniLM-L-6-v2":
            self.client = CrossEncoder(self.model)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def rerank(self, query_text, top_documents):
        pairs = [(query_text, doc) for doc in top_documents]
        scores = self.client.predict(pairs)
        reranked = sorted(zip(top_documents, scores), key=lambda x: x[1], reverse=True)
        return reranked


if __name__ == "__main__":
    from data_loader import DataLoader
    from data_chunker import DataChunker
    from embedding_generator import EmbeddingGenerator
    from vector_store import VectorStore

    data_loader = DataLoader(file_paths=["assets/attention_is_all_you_need.pdf"])
    data = data_loader.load_data()
    data = data_loader.flatten_content(data)

    data_chunker = DataChunker(data, method="sentence", sentences_per_chunk=3)
    chunks = data_chunker.chunk_text()
    print(f"Number of chunks: {len(chunks)}")

    embedding_generator = EmbeddingGenerator(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    embeddings = [embedding_generator.generate(chunk) for chunk in chunks]

    vector_store = VectorStore(collection_name="test_collection")
    vector_store.add_documents(chunks, embeddings)

    user_query = "What is the attention mechanism?"
    query_embedding = embedding_generator.generate(user_query)
    results = vector_store.query(query_embedding)
    reranker = Reranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranked = reranker.rerank(user_query, results["documents"][0])
    for doc, score in reranked:
        print(f"{score:.4f} - {doc[:50]}")
