# DarGlobal and Wasalt Assistant

A retrieval augmented chatbot that answers questions about DarGlobal (luxury real estate developer) and Wasalt (Saudi real estate platform), using data collected from their public web pages. Built for the DarGlobal Junior AI Engineer technical assignment.

**Live demo:** https://YOUR_SERVICE.onrender.com
The free host sleeps when idle, so the first load after a quiet period can take up to a minute.

## How it works

1. **Collection.** Public pages from darglobal.co.uk were crawled with Playwright and BeautifulSoup, honouring robots.txt with a one and a half second delay. Wasalt sits behind a Cloudflare bot challenge, so automated crawling was not attempted past it. The Wasalt content was collected manually from public pages in a normal browser and summarised, with source URLs kept.
2. **Cleaning.** Challenge pages are quarantined, duplicates removed, navigation and cookie text stripped. Output is `data/pages_final.jsonl`.
3. **Indexing.** Pages are split into chunks of about 900 characters with headings attached, then embedded with `BAAI/bge-small-en-v1.5` through fastembed (ONNX, no PyTorch). The index is built during the Docker image build, so build time and runtime use the same library version.
4. **Retrieval.** Each question runs a dense search (cosine similarity) and a BM25 keyword search. Results are fused with reciprocal rank fusion, with at most two chunks per page for diversity. Plain "latest news" questions read a digest of the DarGlobal newsroom, which lists releases newest first with their dates.
5. **Generation.** The top chunks go into a grounded prompt sent to OpenRouter. The app resolves currently free models at runtime and sends a fallback list, ending with `openrouter/free`, so a rotating free model cannot take the app down. Empty answers are retried once.
6. **Safeguards.** The model must answer only from context, never invent figures, and decline off topic questions. Repeated questions are cached, and requests are rate limited per client.

## Run locally

```bash
cp .env.example .env        # then put your OpenRouter key in .env
docker build -t darglobal-bot .
docker run --rm -p 7860:7860 --env-file .env darglobal-bot
```

Open http://localhost:7860

Without Docker:

```bash
pip install -r requirements.txt
python build_index.py
export OPENROUTER_API_KEY=your_key
uvicorn app.main:app --port 7860
```

## Deploy on Render (free Docker web service)

1. Push this folder to a GitHub repository (never commit `.env`).
2. On render.com choose New, then Web Service, and select the repository.
3. Choose the Docker runtime and the Free instance type. The Dockerfile already reads the `PORT` variable Render provides.
4. Add the environment variable `OPENROUTER_API_KEY`. Set the health check path to `/health`.
5. If the 512 MB instance runs out of memory, add `USE_EMBEDDINGS=false` to fall back to keyword search only.

## Configuration

| Variable             | Purpose                                                      | Default                                 |
| -------------------- | ------------------------------------------------------------ | --------------------------------------- |
| `OPENROUTER_API_KEY` | OpenRouter key (required)                                    | none                                    |
| `USE_EMBEDDINGS`     | Set to false to skip the embedding model on low memory hosts | true                                    |
| `PREFERRED_MODELS`   | Comma separated preferred free models                        | Nemotron 3 Ultra, gpt-oss-120b, Gemma 4 |
| `RATE_LIMIT_PER_MIN` | Requests per minute per client                               | 12                                      |

## Limitations

- Free OpenRouter models are rate limited and rotate, so occasional delays or a polite "try again" message are possible.
- The knowledge base is a snapshot from 18 and 19 November 2026. Prices, availability and live listings are not covered.
- Answers are informational and not investment or legal advice.
- The content belongs to the respective companies and is used here only for this assessment.
