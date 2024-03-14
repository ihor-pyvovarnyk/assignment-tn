import click

from assignment.main import chat_cmd, load_repo_cmd


@click.command
@click.option(
    "-r", "--repo-url", default=None, help="Repo URL for cloning. Could be omitted if the repo was already cloned"
)
@click.option(
    "-e",
    "--embeddings",
    default=None,
    help="Embeddings file for finding relevant context information. Could be omitted if the repo wasn't yet loaded",
)
def run(repo_url: str | None, embeddings: str | None) -> None:
    if repo_url:
        embeddings_file_path = load_repo_cmd(repo_url)
    else:
        embeddings_file_path = embeddings
    chat_cmd(embeddings_file_path)


if __name__ == "__main__":
    run()
