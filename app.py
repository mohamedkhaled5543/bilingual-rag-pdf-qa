import re
import faiss
import torch
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
EMB_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

st.set_page_config(page_title="Bilingual PDF Q&A", page_icon="📄")
st.title("📄 Bilingual PDF Q&A")
st.caption("Upload a PDF. Ask in Arabic or English.")


@st.cache_resource
def load_models():
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    emb = SentenceTransformer(EMB_NAME)
    return tok, llm, emb


tokenizer, model, embedder = load_models()


def is_arabic(text):
    return bool(re.search(r"[\u0600-\u06FF]", text))


def generate_text(prompt, max_new_tokens=200):
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def extract_text(file):
    reader = PdfReader(file)
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    for s in ["•", "®", "™"]:
        text = text.replace(s, "")
    return text


def chunk_text(text, size=150, overlap=30):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size - overlap)]


def build_index(chunks):
    emb = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def search(query, index, chunks, k=8):
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _, ids = index.search(q, k)
    return [chunks[i] for i in ids[0]]


def rag_answer(question, index, chunks, k=8):
    ar = is_arabic(question)
    if ar:
        query = generate_text(
            "Translate the following Arabic financial question into a concise English query. "
            "Use exact reporting line items (e.g., 'Total net sales'). "
            f"Output ONLY the translated query.\n\nQuestion: {question}",
            max_new_tokens=40,
        )
    else:
        query = question

    top = search(query, index, chunks, k)
    context = "\n\n".join(top)
    lang = (
        "CRITICAL: Write the entire response in ARABIC (باللغة العربية فقط). "
        "Keep numbers exactly as written in the text."
        if ar else "Write the entire response in English."
    )
    prompt = (
        "Answer the question using ONLY the provided text.\n"
        f"{lang}\n"
        "- Copy numbers EXACTLY. Never add or change digits.\n"
        "- Do NOT sum numbers across categories. Use the exact line-item value.\n"
        "- Include the unit if the text states one.\n"
        "- In tables, the FIRST number is 'Three Months Ended'. Use it for quarter questions.\n"
        "- Answer in one or two sentences.\n"
        "- If not stated, respond 'Not stated in the document' (Arabic: غير مذكور في المستند).\n\n"
        f"Text:\n{context}\n\nQuestion: {question}"
    )
    return generate_text(prompt, max_new_tokens=250), top


uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    if st.session_state.get("file_name") != uploaded.name:
        with st.spinner("Indexing document..."):
            chunks = chunk_text(extract_text(uploaded))
            st.session_state.chunks = chunks
            st.session_state.index = build_index(chunks)
            st.session_state.file_name = uploaded.name
        st.success(f"Ready. {len(st.session_state.chunks)} chunks.")

    question = st.text_input("Ask a question (Arabic or English)")
    if question:
        with st.spinner("Thinking..."):
            answer, sources = rag_answer(
                question, st.session_state.index, st.session_state.chunks
            )
        st.subheader("Answer")
        st.write(answer)
        with st.expander("Sources"):
            for i, s in enumerate(sources[:3], 1):
                st.markdown(f"**Chunk {i}**")
                st.text(s[:600])
