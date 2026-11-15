# PDF Chat

A minimal agentic chat app for asking questions about uploaded PDFs, built with Chainlit and the OpenAI
Responses API. The task said PDFs "vary heavily in size and quality", so most of the design is about
**not treating every PDF the same way**:

- **Quality is decided per page.** Each page is classified as text, scanned, garbled or empty. Scanned and
  garbled pages are OCR'd with a vision model; anything still unreadable can be looked at by the agent on demand.
- **Size is decided per session.** Documents that fit a token budget go into the prompt whole (no retrieval, best
  answers, cheap with prompt caching). Larger ones are chunked and searched with hybrid (semantic + keyword) search.
- **Answers are grounded and checkable.** Every claim cites a page; citations open the PDF at that page, show the
  *printed* page number when it differs from the PDF position, and are flagged when the model cites a page it
  never actually saw.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and an OpenAI API key.

```bash
uv sync
cp .env.example .env          # then set OPENAI_API_KEY
uv run chainlit run app.py -w # http://localhost:8000 (CHAINLIT_PORT to change)
```

Drop PDFs into the chat, then ask questions. The same agent also runs without the UI:

```bash
uv run python scripts/ask.py samples/attention.pdf -q "What BLEU did the big model get?" -q "How long did training take?"
uv run pytest -q              # 83 tests, no network or API key needed
```

### Sample documents

Chosen so each one exercises a different path:

| File | Pages | Exercises |
|---|---|---|
| `attention.pdf`: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) | 15 | small, clean, two-column, fits in the prompt |
| `nist-sp800-53r5.pdf`: [NIST SP 800-53 r5](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf) | 492 | large, retrieval, running headers, roman + arabic page numbers |
| `attention-scanned.pdf` (generated) | 15 | every page image-only and noisy: full OCR |
| `attention-mixed.pdf` (generated) | 15 | digital, with pages 3-4 scanned (one holds Figure 1) |
| Sample Employee Handbook ([National Council of Nonprofits](https://www.501commons.org/resources/tools-and-best-practices/human-resources/sample-employee-handbook-national-council-of-nonprofits)) | 28 | printed page numbers differ from PDF positions |

```bash
curl -L -o samples/attention.pdf https://arxiv.org/pdf/1706.03762
curl -L -o samples/nist-sp800-53r5.pdf https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf
uv run python scripts/make_samples.py   # builds the scanned and mixed variants from attention.pdf
```

The scanned samples are simulated (low DPI, grayscale, noise, JPEG artefacts) because reliable public scans are hard
to source, and because starting from a clean original gives ground truth to measure OCR against.

## How it works

```
 upload
   │
   ▼
 ingest.py ── validate (magic bytes, size, pages, encryption)
   │          extract text per page (PyMuPDF), no API calls
   │          classify each page: text | scanned | garbled | empty
   │          strip running headers/footers, detect printed page numbers
   ▼
 ocr.py ───── scanned/garbled pages → vision model, 8 in parallel, ≤40 pages,
   │          cached by content hash; output goes through the same cleanup
   ▼
 index.py ─── ~400-token chunks (never crossing a page) → embeddings + BM25
   │
   ▼
 agent.py ─── per turn: system prompt = rules + document catalogue
   │            + full text of documents that fit the budget (prompt.py)
   │          loop: model ⇄ tools (search, read_pages, view_page) until it answers
   ▼
 citations.py normalise [file p.N] → printed page numbers, links, unverified flags
   │
   ▼
 app.py ───── Chainlit: streamed answer, tool calls as steps, PDF side panel
```

| Module | Responsibility |
|---|---|
| `pdfchat/config.py` | Every threshold and model choice, with the reasoning next to it |
| `pdfchat/ingest.py` | Validation, extraction, page classification, header stripping |
| `pdfchat/pagelabels.py` | Printed page numbers vs PDF positions |
| `pdfchat/ocr.py` | Vision OCR policy: cap, concurrency, cache, failure handling |
| `pdfchat/index.py` | Chunking, embeddings, hybrid search with Reciprocal Rank Fusion |
| `pdfchat/prompt.py` | System prompt, which documents go in context |
| `pdfchat/tools.py` | The agent's tools |
| `pdfchat/agent.py` | Hand-rolled tool-calling loop; yields UI-agnostic events |
| `pdfchat/citations.py` | Citation normalisation and verification |
| `app.py` | Chainlit glue only |

## Design decisions

**Page classification instead of one extraction path.** Real PDFs are often mixed: a digital report with a scanned
signature page or appendix. Each page is judged on its own: fewer than 50 meaningful characters on a mostly-image page
means scanned; a text layer that is mostly non-prose characters (broken font encodings, `(cid:NN)` glyphs) means
garbled, and its junk text is discarded rather than indexed.

**Small documents skip RAG.** Retrieval is a lossy workaround for limited context. A 10k-token paper is answered
better, and with fewer moving parts, by putting it in the prompt. The 60k-token budget is shared by the session
(three "small" PDFs can still be too much together); documents are admitted in upload order and the rest are
search-only. The prompt is deterministic, so OpenAI's prefix cache makes follow-up turns ~10x cheaper
(measured: 10,746 of 10,805 input tokens cached on the second question, $0.0011 → $0.0001).

**Hybrid search.** Embeddings find paraphrases ("employee leaves" → personnel termination); BM25 finds exact tokens
embeddings blur (control IDs like `AC-2`, names, numbers). Reciprocal Rank Fusion merges the rankings without
calibrating their scores. Chunks never cross page boundaries, so every hit has exactly one page to cite.

**Running header/footer removal.** On NIST, nearly every chunk began with the same title, URL and "PAGE N". Lines
found in the top/bottom 5 lines of at least half the pages are removed (digits normalised); restricting to those
zones keeps repeated *content* such as "Related Controls: None." This moved the defining pages of controls to rank 1.

**Hand-rolled agent loop.** About 100 lines on the Responses API, with no framework to explain or debug. The agent emits
events (text delta, tool started/finished, completed), so it is tested with a scripted fake client and reused by
the CLI. Tool errors go back to the model as text it can correct; the last allowed step disables tools so a turn
always ends with an answer; independent tool calls run in parallel. Between turns only questions and answers are
kept: tool output is per-turn scratch work.

**Stateless requests** (`store=False`, encrypted reasoning carried client-side): uploaded documents are not retained
by OpenAI between requests.

**Three tools, no more.** `search`, `read_pages` (≤6 pages) and `view_page` (page image, for scans, tables,
charts). There is no "list documents" tool because the catalogue is already in the system prompt.

**Citations are verified, not just requested.** The model is asked to cite `[file p.N]`. In testing it described
a figure on a scanned page it had never opened (it knew the famous paper) and cited that page. Prompting alone
didn't reliably prevent that, so the agent records which pages it actually showed the model (in-context text and tool
results), and citations of any other page are marked *(unverified)* in the answer.

**Printed page numbers.** In the employee handbook, the retirement plan is on PDF page 13 but printed page 9; a
reader with a paper copy needs the latter. The model works only with PDF positions (one unambiguous numbering, the
one all tools use); printed numbers are resolved at ingest and added when rendering, e.g.
`[Handbook.pdf p.9 (PDF p.13)]`. Source order: the PDF's own `/PageLabels` (ignored when they just repeat
positions, which many tools write), then numbers printed in the header/footer zone, accepted only as runs of ≥3
consecutive pages with a constant offset. That handles roman front matter and per-chapter restarts, and rejects stray
numbers. The catalogue tells the model the mapping so "page 11 of my printed copy" works.

**OCR with a vision model, eagerly but bounded.** No system dependency (unlike Tesseract), tables come out as
Markdown and figures get a one-line description, which helps retrieval. Up to 40 pages per document are OCR'd at upload,
8 at a time; anything beyond that, or any page whose OCR failed, stays reachable through `view_page`. OCR output keeps
headers and page numbers and then goes through the same cleanup as digital text, so scanned documents get running
header removal and printed page numbers too.

### Model choice (measured, not assumed)

OCR on three pages of the degraded scan, compared against the original text:

| Model | Word recall (3 pages) | Latency / page | Cost / page |
|---|---|---|---|
| **gpt-6-luna** (default) | 99.1 / 94.7 / 98.4 % | 16.4 s | $0.0010 |
| gpt-5-mini (deprecated 2026-12-11) | 99.5 / 96.9 / 98.6 % | 11.6 s | $0.0025 |
| gpt-6.1-terra | 99.8 / 94.7 / 98.6 % | 12.6 s | $0.0068 |

Quality is equivalent (all three drop words on the formula-heavy page), so the cheapest model wins. With
`reasoning.effort = "none"` luna got faster (10.5-11.5 s vs 16-21 s per page) at the same recall, which is the OCR
setting. Agent turns use `low`. gpt-6-luna only supports function calling in Chat Completions with reasoning off,
which is one more reason for the Responses API. `scripts/compare_models.py` reproduces the table.

### What things cost

| Action | Measured |
|---|---|
| Index NIST (492 pages, 1,019 chunks) | ~2 s parse + ~2.5 s embedding, ~$0.007 |
| OCR 15 scanned pages | 39 s wall clock, ~$0.015 |
| Question on a small in-context document | ~$0.001 first turn, ~$0.0001 cached follow-up |
| Question on NIST (search) | ~$0.0005 |
| Question needing `view_page` | ~$0.0015 |

Each answer shows its token count and cost in the UI (`SHOW_USAGE=0` hides it).

## Configuration

Defaults live in `pdfchat/config.py`; the ones worth overriding are environment variables (see `.env.example`):

| Variable | Default | |
|---|---|---|
| `OPENAI_API_KEY` | (required) | |
| `CHAT_MODEL` / `VISION_MODEL` | `gpt-6-luna` | agent / OCR + `view_page` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | |
| `REASONING_EFFORT` / `OCR_REASONING_EFFORT` | `low` / `none` | |
| `SHOW_USAGE` | `1` | token and cost line under answers |
| `CHAINLIT_PORT` / `CHAINLIT_HOST` | `8000` / `127.0.0.1` | read by Chainlit itself |

## Testing

`uv run pytest -q` runs 83 tests in ~3 s without network access. They use PDFs built in memory, a deterministic
hashing embedder, a scripted Responses client for the agent loop, a fake transcriber for OCR orchestration, and a real
Chainlit context for the UI glue. Three bugs were caught on the way that the tests now pin: PyMuPDF happily opening an
HTML error page as an 8-page "PDF", the streaming SDK's `parsed_arguments` being rejected when echoed back, and
citation links failing Chainlit's element validation.

Live behaviour is checked with the scripts: `ask.py` (end to end), `search_demo.py` (what each retriever
contributed per hit) and `compare_models.py`.

## Limitations and what I'd do next

Roughly in the order I'd tackle them:

- **Evaluation.** Build a small golden set (questions, expected answers, expected pages) per sample type and track
  retrieval recall@k and answer faithfulness, then tune thresholds (chunk size, budget, classification cut-offs) against
  it instead of by eye.
- **Persistence and scale.** Everything is in memory per chat session (except the OCR cache, which is per process). Next:
  store documents, chunks and embeddings (Postgres + pgvector, or Qdrant) keyed by content hash, behind the same
  `Index.search` signature.
- **Background ingestion.** Parsing and OCR run inside the upload request. Fine for tens of pages; for hundreds of scanned
  pages they belong in a job queue, with the document becoming queryable progressively.
- **Layout-aware parsing.** PyMuPDF's text order is usually right but tables lose their structure and complex layouts can
  interleave. Docling, Marker or Unstructured would give Markdown with tables and headings, which would also allow
  section-based citations ("IX.D Retirement Plan"), as useful to a reader as a page number.
- **Visual retrieval.** For figure- and table-heavy documents, retrieve page *images* (ColPali-style) instead of, or as
  well as, text chunks.
- **Managed alternatives considered.** OpenAI File Search or passing the PDF directly as model input would be the
  zero-code baseline. I didn't build on them because they hide the parts this task is about (OCR policy, routing,
  page-accurate and verified citations) and offer no control over cost per document. They'd be the first comparison point in an eval.
- **Security.** File size, page count, encryption and magic bytes are checked, but PDF parsing runs in-process; in
  production it belongs in a sandboxed worker. Prompt injection from document content is only mitigated by the system
  prompt (documents are data, not instructions) and by the agent having no side-effecting tools.
- **Smaller gaps.** Chat history grows without summarisation; a numbered run shorter than three pages gets no printed
  page number; a chunk's overlap does not span page boundaries.
