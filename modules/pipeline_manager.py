from modules.data_loader import DataLoader
from modules.data_chunker import DataChunker
from modules.embedding_generator import EmbeddingGenerator
from modules.vector_store import VectorStore
from modules.reranker import Reranker
from modules.prompt_constructor import PromptConstructor
from modules.llm import LLM


class PipelineManager:
    def __init__(self, run_id, config):
        self.run_id = run_id
        self.config = config
        self.data_loader = DataLoader(file_paths=self.config.file_paths)
        self.data_chunker = None
        self.embedding_generator = None
        self.vector_storage = None
        self.reranker = None
        self.prompt_constructor = None
        self.llm_inference = None

    def run(self):
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
        contexts = [doc for doc, score in reranked[:3]]
        context_str = "\n\n -- \n\n".join(contexts)
        prompt_template = "Given the following contexts, answer the question:\n\nContexts:\n -- \n\n{contexts}\n\nQuestion: {question}\n\nAnswer:"

        prompt_constructor = PromptConstructor(template=prompt_template)
        final_prompt = prompt_constructor.construct(
            contexts=context_str, question=user_query
        )

        llm = LLM(model_name="gpt-4.1-mini")
        system_prompt = "You are a helpful assistant that provides concise and accurate answers based on the provided contexts."
        answer = llm.generate(final_prompt, system_prompt=system_prompt)

        return answer


if __name__ == "__main__":
    from config import Config
    from datetime import datetime, timezone
    import uuid

    config = Config(
        file_paths=[
            "../assets/attention_is_all_you_need.pdf",
            # "../assets/attention_is_all_you_need.txt",
            # "../assets/attention_is_all_you_need.docx",
        ],
        method="character",
        chunk_size=1500,
        model_name="gpt-4o-mini",
        n_questions_per_chunk=2,
    )
    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    pipeline_manager = PipelineManager(run_id, config)
    model_response = pipeline_manager.run()
    print(f"Model Response: {model_response}")
