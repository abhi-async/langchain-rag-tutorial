from langchain_core.embeddings import Embeddings
from gpt4all import Embed4All


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2.gguf2.f16.gguf"):
        self.model = Embed4All(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed(text)