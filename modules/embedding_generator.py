class EmbeddingGenerator:
    def __init__(self, model):
        self.model = model
        self.client = self.load_model()

    def load_model(self):
        if self.model == "text-embedding-3-small":
            from openai import OpenAI

            client = OpenAI()
        elif self.model == "sentence-transformers/all-mpnet-base-v2":
            from sentence_transformers import SentenceTransformer

            client = SentenceTransformer("all-mpnet-base-v2")

        elif self.model == "sentence-transformers/all-MiniLM-L6-v2":
            from sentence_transformers import SentenceTransformer

            client = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            raise ValueError(f"Unsupported model: {self.model}")
        return client

    def generate(self, text):
        if self.model == "text-embedding-3-small":
            response = self.client.embeddings.create(input=text, model=self.model)
        elif self.model == "sentence-transformers/all-MiniLM-L6-v2":
            response = self.client.encode(text)
        elif self.model == "sentence-transformers/all-mpnet-base-v2":
            response = self.client.encode(text)
        return response


if __name__ == "__main__":
    from data_loader import DataLoader
    from data_chunker import DataChunker

    data_loader = DataLoader(file_paths=["assets/attention_is_all_you_need.pdf"])
    data = data_loader.load_data()
    data = data_loader.flatten_content(data)

    data_chunker = DataChunker(data, method="sentence", sentences_per_chunk=3)
    chunks = data_chunker.chunk_text()
    print(f"Number of chunks: {len(chunks)}")

    embedding_generator = EmbeddingGenerator(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    for chunk in chunks:
        embedding = embedding_generator.generate(chunk)
        print(f"Embedding Length: {len(embedding)}\n")
