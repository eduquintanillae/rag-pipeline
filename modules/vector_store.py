import chromadb


class VectorStore:
    def __init__(self, collection_name):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=collection_name)

    def add_documents(self, documents, embeddings):
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=[str(i) for i in range(len(documents))],
        )
        print(
            f"Added {len(documents)} documents to the collection '{self.collection.name}'."
        )

    def query(self, query_embedding, n_results=5):
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=n_results
        )
        return results


if __name__ == "__main__":
    from data_loader import DataLoader
    from data_chunker import DataChunker
    from embedding_generator import EmbeddingGenerator

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
    query_embedding = embedding_generator.generate("What is the attention mechanism?")
    results = vector_store.query(query_embedding)
    print(f"Query results: {results}")
