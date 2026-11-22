import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Iterator, List

import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==============================================================================
# LOCAL STORAGE DIRECTORIES (Persists models & FAISS indexes across restarts)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
INDEX_CACHE_DIR = BASE_DIR / "faiss_cache"
MODELS_DIR.mkdir(exist_ok=True)
INDEX_CACHE_DIR.mkdir(exist_ok=True)


# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Local HR Policy RAG",
    page_icon="📑",
    layout="wide",
)

st.title("📑 Local HR Policy RAG Assistant")
st.caption("100% Offline Inference • Qwen3-Embedding-0.6B (CPU) + Qwen3-4B-GGUF Q4_K_M (llama.cpp)")


# ==============================================================================
# 1. LOCAL HARDWARE-AWARE MODEL LOADERS
# ==============================================================================
@st.cache_resource(show_spinner="Loading local embedding model (CPU)...")
def get_embedding_model() -> HuggingFaceEmbeddings:
    # Kept on CPU intentionally: embedding a 15-page PDF takes ~1s and saves VRAM for the LLM
    return HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        cache_folder=str(MODELS_DIR),
        model_kwargs={
            "device": "cpu",
            "trust_remote_code": True,
        },
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        query_encode_kwargs={
            "prompt_name": "query",
            "normalize_embeddings": True,
        },
    )


@st.cache_resource(show_spinner="Initializing Qwen3-4B-GGUF engine...")
def get_llm(n_gpu_layers: int, n_threads: int) -> LlamaCpp:
    local_gguf = MODELS_DIR / "Qwen3-4B-Q4_K_M.gguf"

    # Download once into ./models; skip network check on subsequent local runs
    if not local_gguf.exists():
        hf_hub_download(
            repo_id="Qwen/Qwen3-4B-GGUF",
            filename="Qwen3-4B-Q4_K_M.gguf",
            local_dir=str(MODELS_DIR),
        )

    return LlamaCpp(
        model_path=str(local_gguf),
        n_ctx=4096,               # Lean 4K context window minimizes local RAM/KV-cache footprint
        n_batch=512,
        n_threads=n_threads,      # Matched to physical cores to prevent thread thrashing
        n_gpu_layers=n_gpu_layers,
        max_tokens=512,
        temperature=0.0,
        repeat_penalty=1.08,
        streaming=True,           # Enables live token-by-token UI updates
        stop=["<|im_end|>"],
        verbose=False,
    )


# ==============================================================================
# 2. DISK-PERSISTED FAISS VECTOR STORE
# ==============================================================================
def parse_and_chunk_pdf(pdf_bytes: bytes) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        raw_pages = loader.load()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    current_category = "General Policy"
    enriched_pages: List[Document] = []

    for doc in raw_pages:
        text = doc.page_content
        category_matches = re.findall(r"Policy Category:\s*([A-Za-z\s]+)", text)
        if category_matches:
            current_category = category_matches[-1].strip().splitlines()[0]

        doc.metadata["policy_category"] = current_category
        doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1
        enriched_pages.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=120,
        separators=[
            r"\nPolicy Category:",
            r"\n\d+\.\s+",
            "\n\n",
            "\n",
            " ",
            "",
        ],
        is_separator_regex=True,
    )

    chunks = splitter.split_documents(enriched_pages)
    for chunk in chunks:
        cat = chunk.metadata.get("policy_category", "General")
        pg = chunk.metadata.get("page_number", "?")
        chunk.page_content = f"[Category: {cat} | Page: {pg}]\n{chunk.page_content.strip()}"

    return chunks


def get_or_build_vectorstore(pdf_bytes: bytes, file_hash: str) -> tuple[FAISS, int, bool]:
    """Loads FAISS index from ./faiss_cache if available; otherwise builds and saves it."""
    if st.session_state.get("indexed_file_hash") == file_hash:
        return st.session_state["vectorstore"], st.session_state["chunk_count"], True

    embeddings = get_embedding_model()
    disk_index_path = INDEX_CACHE_DIR / file_hash

    if disk_index_path.exists():
        vectorstore = FAISS.load_local(
            str(disk_index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        chunk_count = vectorstore.index.ntotal
        from_disk = True
    else:
        with st.spinner("Chunking PDF and indexing locally (saved to ./faiss_cache)..."):
            chunks = parse_and_chunk_pdf(pdf_bytes)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(str(disk_index_path))
            chunk_count = len(chunks)
            from_disk = False

    st.session_state["vectorstore"] = vectorstore
    st.session_state["chunk_count"] = chunk_count
    st.session_state["indexed_file_hash"] = file_hash
    return vectorstore, chunk_count, from_disk


# ==============================================================================
# 3. STREAMING RAG GENERATOR
# ==============================================================================
def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def stream_rag_response(
    vectorstore: FAISS,
    llm: LlamaCpp,
    query: str,
    k_chunks: int,
) -> tuple[Iterator[str], List[Document]]:
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k_chunks, "fetch_k": k_chunks * 3, "lambda_mult": 0.7},
    )

    retrieved_docs = retriever.invoke(query)
    context_str = format_docs(retrieved_docs)

    prompt = ChatPromptTemplate.from_template(
        """<|im_start|>system
You are an objective HR Policy Research Analyst. Answer the employee's question strictly using ONLY the provided HR policy excerpts.

Rules:
1. Cite the exact Policy Category and Page number for every claim (e.g., [Leave Policy, Page 5]).
2. Highlight any cross-policy conditions, exceptions, or numerical thresholds (days, currency limits, deadlines).
3. If the provided context does not contain the answer, state: "The provided HR policy document does not specify this information." Do not extrapolate.
<|im_end|>
<|im_start|>user
Context Excerpts:
{context}

Question: {question} /no_think
<|im_end|>
<|im_start|>assistant
"""
    )

    chain = prompt | llm

    def token_generator() -> Iterator[str]:
        in_think_block = False
        for chunk in chain.stream({"context": context_str, "question": query}):
            # Suppress any stray <think> tags during live streaming
            if "<think>" in chunk:
                in_think_block = True
                continue
            if "</think>" in chunk:
                in_think_block = False
                continue
            if not in_think_block:
                yield chunk

    return token_generator(), retrieved_docs


# ==============================================================================
# 4. SIDEBAR: LOCAL HARDWARE TUNING & PDF UPLOAD
# ==============================================================================
physical_cores = max(1, (os.cpu_count() or 4) // 2)
has_cuda = torch.cuda.is_available()

with st.sidebar:
    st.header("📂 Document & Hardware Tuning")
    uploaded_file = st.file_uploader("Upload HR Policy PDF", type=["pdf"])

    k_chunks = st.slider("Retrieved Chunks (MMR k)", min_value=2, max_value=7, value=4)

    with st.expander("⚙️ Local Compute Settings", expanded=False):
        n_threads = st.slider(
            "CPU Threads (Physical Cores)",
            min_value=1,
            max_value=os.cpu_count() or 8,
            value=physical_cores,
            help="Set to your physical core count (typically half of logical processors).",
        )
        n_gpu_layers = st.slider(
            "GPU Offload Layers",
            min_value=0,
            max_value=36,
            value=36 if has_cuda else 0,
            help="0 = pure CPU mode. Increase only if llama-cpp-python was compiled with CUDA/Vulkan.",
        )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    st.divider()
    st.caption(f"Local Cache: `{MODELS_DIR.name}/` & `{INDEX_CACHE_DIR.name}/`")


# ==============================================================================
# 5. MAIN CHAT INTERFACE
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if uploaded_file is None:
    st.info("👈 Upload your HR Policy PDF in the sidebar to start.")
    st.stop()

pdf_bytes = uploaded_file.getvalue()
file_hash = hashlib.md5(pdf_bytes).hexdigest()
vectorstore, chunk_count, loaded_from_disk = get_or_build_vectorstore(pdf_bytes, file_hash)

cache_badge = "loaded from disk cache" if loaded_from_disk else "newly indexed"
st.sidebar.success(f"**{uploaded_file.name}** ({chunk_count} clauses • {cache_badge})")

llm = get_llm(n_gpu_layers=n_gpu_layers, n_threads=n_threads)

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("🔍 View Retrieved Policy Excerpts"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {idx}:**")
                    st.code(src, language="text")

if user_query := st.chat_input("Ask a question about leave, notice period, reimbursements, WFH..."):
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        token_stream, docs = stream_rag_response(vectorstore, llm, user_query, k_chunks)
        full_answer = st.write_stream(token_stream)

        source_texts = [d.page_content for d in docs]
        with st.expander("🔍 View Retrieved Policy Excerpts"):
            for idx, src in enumerate(source_texts, 1):
                st.markdown(f"**Chunk {idx}:**")
                st.code(src, language="text")

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": full_answer,
            "sources": source_texts,
        }
    )