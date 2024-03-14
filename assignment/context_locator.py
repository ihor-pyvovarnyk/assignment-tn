import numpy as np
import pandas as pd
from openai import OpenAI

from assignment.embeddings_utils import distances_from_embeddings


class ContextLocator:
    EMBEDDINGS_MODEL = "text-embedding-3-small"
    MAX_DISTANCE_COSINE = 0.8
    MAX_CONTEXT_SIZE = 10

    def __init__(self, client: OpenAI, embeddings_file: str):
        self.client = client
        self.embeddings_file = embeddings_file

    def find_context(self, prompt: str) -> list[dict]:
        emb = self.client.embeddings.create(input=prompt, model=self.EMBEDDINGS_MODEL)
        df = pd.read_csv(self.embeddings_file)
        df["embedding"] = df.embedding.apply(eval).apply(np.array)  # type: ignore
        distances = distances_from_embeddings(emb.data[0].embedding, df["embedding"].values, distance_metric="cosine")
        mask = (np.array(distances) < self.MAX_DISTANCE_COSINE)[np.argsort(distances)]
        messages = list(df["message"].iloc[np.argsort(distances)][mask])
        messages = messages[: self.MAX_CONTEXT_SIZE] if messages else []

        if messages:
            return [
                {"role": "system", "content": f"Knowledge: {message}\nOnly answer user prompts."}
                for message in messages
            ]
        return []
