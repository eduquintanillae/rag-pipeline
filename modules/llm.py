from openai import OpenAI
import dotenv
import os
from openai import OpenAI
import re
import time

dotenv.load_dotenv()


class LLM:
    def __init__(self, model_name):
        self.model_name = model_name

        if self.model_name == "gpt-4.1-mini":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt, system_prompt=None):
        initial_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
        )
        total_time = time.time() - initial_time
        model_response = response.choices[0].message.content.strip()

        results = {
            "model_response": model_response,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "total_time": total_time,
        }

        return results


if __name__ == "__main__":
    from data_loader import DataLoader
    from data_chunker import DataChunker
    from embedding_generator import EmbeddingGenerator
    from vector_store import VectorStore
    from reranker import Reranker
    from prompt_constructor import PromptConstructor

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
    print(f"Generated Answer: {answer}")
