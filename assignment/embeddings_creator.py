import csv
import json
import os
from functools import cached_property
from math import ceil
from typing import Iterator
from uuid import uuid4

import click
from openai import OpenAI


class EmbeddingsCreator:
    EMBEDDINGS_DIR = "./embeddings/"
    EMBEDDINGS_MODEL = "text-embedding-3-small"
    HEADER_EMBEDDING = "embedding"
    HEADER_MESSAGE = "message"
    HEADERS = [HEADER_EMBEDDING, HEADER_MESSAGE]
    MAX_CONTENT_SIZE = 8000

    def __init__(self, client: OpenAI, repo_dir: str):
        self.client = client
        self.repo_dir = repo_dir

    @cached_property
    def output_file_path(self) -> str:
        return os.path.join(self.EMBEDDINGS_DIR, f"{uuid4()}.csv")

    def create(self) -> str:
        with open(self.output_file_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.HEADERS)
            writer.writeheader()

            for file_path, content in self._files_generator():

                for content_part in self._split_content(content):
                    input_content = f"File: {file_path}\nContent: {content_part}"
                    emb = self.client.embeddings.create(input=input_content, model=self.EMBEDDINGS_MODEL)
                    writer.writerow(
                        {
                            self.HEADER_EMBEDDING: json.dumps(emb.data[0].embedding),
                            self.HEADER_MESSAGE: input_content,
                        }
                    )

        return self.output_file_path

    def _files_generator(self) -> Iterator[str]:
        for root, _, files in os.walk(self.repo_dir):
            if any(d.startswith(".") and d != "." for d in root.split("/")):  # Skip dot dirs
                continue
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, "r") as file:
                        content = file.read()
                    yield file_path, content
                    click.echo(f"Processed file {file_path}")
                except Exception as e:
                    click.echo(f"Skipped file {file_path}: {e}")

    def _split_content(self, content: str) -> Iterator[str]:
        # Not the most elegant approach, but need to somehow split
        # larger files since they can't be turned into embeddings
        for i in range(ceil(len(content) / self.MAX_CONTENT_SIZE)):
            start_idx = i * self.MAX_CONTENT_SIZE
            end_idx = start_idx + self.MAX_CONTENT_SIZE
            yield content[start_idx:end_idx]
