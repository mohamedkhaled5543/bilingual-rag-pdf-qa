# Bilingual PDF Q&A (Arabic + English)

Ask questions about a PDF in Arabic or English. Get answers grounded in the document, with sources.

Built on Apple's Q3 2026 10-Q filing.

## Demo
![demo](demo.gif)

## Problem
Financial filings are long and English-only. Finding one number takes minutes.

## Solution
A RAG system. Ask in Arabic or English. Get the exact figure and the source chunk.

## Architecture
PDF → clean text → chunk (150 words) → multilingual embeddings → FAISS → Mistral-7B → answer

Arabic questions are translated to an English retrieval query first. The answer is written in the question's language.

## Stack
- Mistral-7B-Instruct-v0.3
- paraphrase-multilingual-MiniLM-L12-v2
- FAISS (cosine similarity)
- PyPDF2
- Streamlit

## Results
6 test questions (3 EN, 3 AR) on the 10-Q.

| Question | Expected | Result |
|---|---|---|
| Total net sales (quarter) | $109,417M | ✅ |
| iPhone net sales (quarter) | $54,252M | ✅ |
| Diluted EPS (quarter) | $2.02 | ✅ |

## What I fixed along the way
- Symbols like `®` broke retrieval. Cleaned them.
- Large chunks mixed quarter and nine-month columns. Used smaller chunks.
- Model grabbed the wrong column. Added a table-reading rule to the prompt.
- Questions now use exact line-item names.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Needs a GPU. See `run_colab.md` for the Colab/Kaggle setup.

## Files
- `app.py`: Streamlit app
- `rag-system.ipynb`: notebook version
- `sample/10Q.pdf`: test document

## Next
- Hybrid search (BM25 + embeddings)
- Multi-PDF support
- API model for cheap deployment

## Author
Mohamed (ZOREN). Building AI automation systems.
