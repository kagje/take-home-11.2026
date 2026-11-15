"""Central place for tunables. Every threshold here is a judgement call worth
discussing; they are deliberately not buried in the code that uses them."""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Models -------------------------------------------------------------------
# gpt-6-luna: $0.10 / $0.50 per 1M in/out tokens, vision + tools, 1M context.
# (gpt-5-mini is deprecated, shutdown 2026-12-11.) For harder documents,
# gpt-6.1-terra ($1 / $5) is OpenAI's recommended gpt-5-mini replacement.
# NOTE: luna only supports function calling in Chat Completions with
# reasoning_effort="none", so the agent uses the Responses API instead.
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-6-luna")
VISION_MODEL = os.getenv("VISION_MODEL", CHAT_MODEL)  # used for OCR + view_page
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Upload guards ------------------------------------------------------------
MAX_FILE_MB = 100
MAX_PAGES = 2_000

# --- Page quality classification ---------------------------------------------
# A page with fewer meaningful characters than this has no usable text layer.
MIN_TEXT_CHARS = 50
# Share of "normal" characters (letters, digits, whitespace, common punctuation)
# below which we treat the text layer as garbage (broken font encodings, etc.).
MIN_PRINTABLE_RATIO = 0.75
# If images cover at least this share of a page with little text -> it's a scan.
SCAN_IMAGE_COVERAGE = 0.5

# --- Running header/footer removal ---------------------------------------------
RUNNING_LINE_MIN_PAGES = 4  # too few pages to tell a header from content
RUNNING_LINE_ZONE = 5  # lines from the top / bottom of a page considered
RUNNING_LINE_MIN_SHARE = 0.5  # must appear on >= this share of pages

# --- Routing ------------------------------------------------------------------
# Documents whose total text fits in this many tokens are given to the model in
# full (no retrieval). Above it we switch to chunk + embed + hybrid search.
FULL_CONTEXT_TOKEN_BUDGET = 60_000

# --- Retrieval ----------------------------------------------------------------
# ~400-token chunks are small enough to be specific, big enough to hold a full
# paragraph or control description. Chunks never cross page boundaries so every
# hit has exactly one page to cite.
CHUNK_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60
EMBED_BATCH = 256  # inputs per embeddings request
SEARCH_TOP_K = 8
SEARCH_CANDIDATES = 50  # per retriever, before fusion
RRF_K = 60  # standard constant from the Reciprocal Rank Fusion paper

# --- OCR ----------------------------------------------------------------------
# Hard cap on vision-OCR pages per document, to bound cost and latency.
MAX_OCR_PAGES = 40
OCR_DPI = 150
