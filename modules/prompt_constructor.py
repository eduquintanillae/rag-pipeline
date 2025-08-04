class PromptConstructor:
    def __init__(self, template):
        self.template = template

    # Constructs a prompt using the provided parameters (could vary based on prompt template)
    def construct(self, **kwargs):
        return self.template.format(**kwargs)


if __name__ == "__main__":
    from data_loader import DataLoader
    from data_chunker import DataChunker
    from embedding_generator import EmbeddingGenerator
    from vector_store import VectorStore
    from reranker import Reranker

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
    relevant_chunks = vector_store.query(query_embedding)

    reranker = Reranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranked = reranker.rerank(user_query, relevant_chunks["documents"][0])

    prompt_template = "Given the following contexts, answer the question:\n\nContexts:\n -- \n\n{contexts}\n\nQuestion: {question}\n\nAnswer:"
    contexts = [doc for doc, score in reranked[:3]]
    context_str = "\n\n -- \n\n".join(contexts)
    prompt_constructor = PromptConstructor(template=prompt_template)
    final_prompt = prompt_constructor.construct(
        contexts=context_str, question=user_query
    )
    print(final_prompt)
