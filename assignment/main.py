import os
import shutil

import click
from git import Repo
from openai import OpenAI

from assignment.context_locator import ContextLocator
from assignment.embeddings_creator import EmbeddingsCreator

REPO_TARGET_DIR = "./temp/repo"
MODEL = "gpt-3.5-turbo"


def load_repo_cmd(repo_url: str) -> str:
    client = _create_openai_client()

    click.echo(f"Loading the repo {repo_url}...")
    if os.path.exists(REPO_TARGET_DIR):
        shutil.rmtree(REPO_TARGET_DIR)
    repo = Repo.clone_from(repo_url, REPO_TARGET_DIR)
    repo_source = repo.working_dir

    click.echo("Pre-processing the repo...")
    embeddings_file_path = EmbeddingsCreator(client, repo_source).create()
    click.echo(f"Created embeddings to reference for chatting: {embeddings_file_path}")
    return embeddings_file_path


def chat_cmd(embeddings_file_path: str) -> None:
    client = _create_openai_client()

    click.echo("You can now enter prompts about the repo (type `exit` to finish)")
    context_locator = ContextLocator(client, embeddings_file_path)

    while True:
        user_prompt = click.prompt("Question")  # e.g. What license do we use?
        if user_prompt == "exit":
            break

        context_prompts = context_locator.find_context(user_prompt)
        resp = client.chat.completions.create(
            messages=[*context_prompts, {"role": "user", "content": user_prompt}],
            model=MODEL,
        )
        click.echo(f"Answer: {resp.choices[0].message.content}")


def _create_openai_client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
