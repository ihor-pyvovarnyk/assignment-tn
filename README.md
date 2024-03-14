# Repo chatter

## Install

```bash
make activate_venv
make install
```

## Run

The following command will clone repo, pre-process it, output a path to an embeddings 
file for looking up relevant information, and start the chat loop:
```bash
OPENAI_API_KEY=<...> python cli.py --repo-url=https://github.com/codota/tabnine-vscode.git
```

The following command will start a chat loop for already processed repo using its 
embeddings as a knowledge base: 
```bash
OPENAI_API_KEY=<...> python cli.py --embeddings=./embeddings/<file-name>.csv
```
