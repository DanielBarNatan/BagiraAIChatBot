# DevOps AI Knowledge Assistant (POC)

Local RAG-based chatbot that answers questions using Azure DevOps Wiki and Work Items (PBI).  
Answers are grounded only in retrieved documents.

## Quick Start (Step-by-Step)

Follow these steps in order to get the bot running from scratch:

1. **Install Python 3.10+** — download from [python.org](https://www.python.org/downloads/) if not already installed.

2. **Clone the repository**

   ```bash
   git clone https://github.com/DanielBarNatan/BagiraAIChatBot.git
   cd BagiraAIChatBot
   ```

3. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # macOS/Linux
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**

   - Copy `.env.example` to `.env`
   - Fill in your values:
     - `OPENAI_API_KEY` — get one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
     - `AZURE_DEVOPS_PAT` — create at [dev.azure.com](https://dev.azure.com) (needs **Wiki Read** + **Work Items Read** permissions)
     - `AZURE_DEVOPS_ORG` — your org name (from `dev.azure.com/{org}/...`)
     - `AZURE_DEVOPS_PROJECT` — the project containing your Wiki and PBIs

6. **Fetch data from Azure DevOps**

   ```bash
   python scripts/fetch_wiki.py
   python scripts/fetch_work_items.py
   ```

7. **Clean the fetched documents**

   ```bash
   python scripts/clean_documents.py
   ```

8. **Chunk the documents**

   ```bash
   python scripts/chunk_documents.py
   ```

9. **Generate embeddings and build the vector index**

   ```bash
   python scripts/generate_embeddings.py
   ```

10. **Run the chatbot**

    ```bash
    python main.py
    ```

    Ask questions in the terminal. Type `quit` or press Enter on an empty line to exit.

---

## Requirements

- Python 3.10+
- [OpenAI API key](https://platform.openai.com/api-keys)
- [Azure DevOps PAT](https://dev.azure.com) with **Wiki (Read)** and **Work Items (Read)**

## Setup

1. **Create virtual environment and install dependencies**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. **Configure environment**

   Copy `.env.example` to `.env` and set:

   - `OPENAI_API_KEY` — your OpenAI API key
   - `AZURE_DEVOPS_PAT` — Azure DevOps Personal Access Token
   - `AZURE_DEVOPS_ORG` — organization name (from `dev.azure.com/{org}/...`)
   - `AZURE_DEVOPS_PROJECT` — project name (Wiki and PBIs)

## Pipeline (run from project root)

Run in order to populate the knowledge base and start chatting:

1. **Fetch data from Azure DevOps**

   ```bash
   python scripts/fetch_wiki.py
   python scripts/fetch_work_items.py
   ```

   Output: `data/raw/wiki/*.md`, `data/raw/pbi/*.txt`

2. **Clean documents**

   ```bash
   python scripts/clean_documents.py
   ```

   Output: `data/cleaned/wiki/`, `data/cleaned/pbi/`

3. **Chunk documents**

   ```bash
   python scripts/chunk_documents.py
   ```

   Output: `data/chunks/chunks.jsonl`

4. **Generate embeddings and build vector index**

   ```bash
   python scripts/generate_embeddings.py
   ```

   Output: `vector_db/chroma_store/` (Chroma DB)

5. **Run the chat**

   ```bash
   python main.py
   ```

   Type questions; type `quit` or press Enter to exit.

## Example questions

- "What is the MissionTrainer architecture?"
- "How does the Multi Drill feature work?"
- "What components exist in the simulator system?"
- "How is the system deployed?"

The assistant answers only from retrieved Wiki and PBI content. If the answer is not in the context, it will say it does not know.

## Project structure

- `config/settings.py` — environment and paths
- `scripts/` — fetch_wiki, fetch_work_items, clean_documents, chunk_documents, generate_embeddings
- `chatbot/` — retriever, prompt_builder, chat_engine
- `main.py` — CLI chat loop
- `data/raw/`, `data/cleaned/`, `data/chunks/` — pipeline data
- `vector_db/chroma_store/` — Chroma vector DB

## Tests

From project root (with venv activated):

```bash
pip install pytest
python -m pytest tests/ -v
```

## Optional env vars

- `AZURE_DEVOPS_WIKI_ID` — set if using a non-default wiki
- `OPENAI_EMBEDDING_MODEL` — default `text-embedding-3-small`
- `OPENAI_CHAT_MODEL` — default `gpt-4o`
- `RETRIEVER_TOP_K` — number of chunks to retrieve (default `15`)


root: cd C:\Users\daniel.bar-natan\Desktop\Bagira chat bot
venv\Scripts\activate
streamlit run app.py