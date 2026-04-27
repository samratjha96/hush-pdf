# hush

Black out names, emails, addresses, and account numbers in your PDF — without sending it to a cloud LLM.

```
+--------+      +---------------+      +---------+
| Drop a | ---> | Review what's | ---> | Save a  |
| PDF    |      | flagged       |      | clean   |
|        |      |               |      | copy    |
+--------+      +---------------+      +---------+
   |                                       ^
   +-------- never touches a third party --+
```

## Quick start (run it yourself, strongest privacy)

```bash
git clone https://github.com/samratjha96/hush-pdf
cd hush-pdf
uv run uvicorn app:app --host 127.0.0.1 --port 8765
```

Then open **http://127.0.0.1:8765** in your browser. The PDF never leaves your machine.

You need [`uv`](https://docs.astral.sh/uv/). One-line install:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The first run downloads `openai/privacy-filter` (~800 MB). After that it's cached locally.

## Quick start (Docker, for VPS deployment)

```bash
docker compose up -d
```

Container exposes port `8765`. Put it behind a reverse proxy (Caddy, Traefik, nginx) for HTTPS.

---

## TL;DR

**The problem.** You have a PDF with stuff in it that shouldn't leak — names of innocent third parties, your home address on a lease, an account number on a statement, an SSN on a tax form. Online "redact PDF" tools want you to upload the file to their server and quietly send the contents through OpenAI / Anthropic / Google Cloud APIs. Drawing black rectangles in Preview leaves the underlying text bytes intact (anyone can copy-paste them out). Adobe Acrobat costs $20+/month and pushes documents through their cloud anyway.

**The solution.** hush detects sensitive text using a small open model that runs either in your browser tab (WebGPU) or on a single self-hosted server you control. Detection never goes to a cloud LLM API. You eyeball every candidate, and hush strips the actual bytes from the PDF when you accept.

| | |
|---|---|
| **No cloud LLM APIs** | Detection runs on `openai/privacy-filter`, a 100 MB token-classification model — not GPT-4, not Claude, not Gemini. Your PDF text is never sent to any third-party AI service. |
| **No third-party storage** | The PDF goes to one server only — either your own laptop, or the single instance you're using. Nothing is copied to S3, Azure Blob, or a CDN. |
| **Deleted after processing** | The hosted backend keeps your file in memory only, never writes it to disk, and clears it the moment you download the cleaned copy (or 15 minutes later, whichever comes first). |
| **Real text deletion** | Accepted regions remove bytes from the PDF's text layer — not paint a black rectangle on top. Copy-paste from the cleaned file produces nothing. |
| **You're the editor** | Every candidate is shown in context. You confirm individually or in bulk. |

---

## Where does your file actually go?

This depends on how you use hush. Pick the row that matches you.

| | Browser (WebGPU) | Server fallback | Self-hosted |
|---|---|---|---|
| **Who runs it?** | Chrome / Edge with WebGPU on a hosted instance | Safari / Firefox / older Chrome on a hosted instance | You, on `127.0.0.1` or your own VPS |
| **Where is the PDF processed?** | Inside your browser tab | On the hush server you connected to | On your machine |
| **Does the file leave your computer?** | No | Yes — it's uploaded to the hush server | No (or: only to your own server) |
| **How long does the server keep it?** | N/A | In RAM only, until you download the cleaned PDF, or 15 min idle | Same — but it's your server |
| **Does any third party see it?** | No | No | No |
| **Cloud LLM APIs involved?** | No, never | No, never | No, never |

The strongest privacy guarantee is "self-hosted" — the file genuinely never leaves your network. The next-strongest is "browser (WebGPU)" — the file stays in your tab even on a hosted instance. "Server fallback" still avoids any third party, but the bytes do touch the server you're using.

---

## How it works

1. You drop a PDF on the page.
2. hush extracts the text layer (in-browser via mupdf-wasm, or server-side via pymupdf).
3. Each chunk runs through `openai/privacy-filter`, a token-classification model that tags PII categories.
4. Candidates appear grouped by category, with bounding boxes drawn over a page preview.
5. You uncheck false positives, draw boxes on missed items, then click **Apply**.
6. hush rewrites the PDF with the accepted regions actually removed from the text layer.
7. You download the cleaned copy. The server-side copy of your PDF (if any) is dropped from memory.

---

## Why "hush"

Four design decisions, in order of importance:

1. **No cloud LLM APIs, ever.** Detection is a small dedicated PII model, not a general-purpose chat model. Your document never becomes prompt context for a third-party AI.
2. **Real redaction, not paint-over.** A black rectangle drawn on top of "Jane Smith" still has "Jane Smith" in the PDF's underlying text. hush deletes the bytes. Copy-paste from the cleaned file produces nothing where the redaction was.
3. **You confirm every redaction.** PII detectors have false positives. The whole point of a redactor is that *you*, not the model, are the final authority on what gets removed.
4. **Two engines, one UI.** Browsers with WebGPU run the model locally via [Transformers.js](https://github.com/huggingface/transformers.js). Browsers without WebGPU (Safari, older Chrome) hand off to the server. The user-facing flow is identical either way.

---

## How hush compares

| | hush | Adobe Acrobat | smallpdf / ilovepdf | Black rectangles in Preview |
|---|:-:|:-:|:-:|:-:|
| Does not call cloud LLM APIs | yes | unclear (closed source) | unclear (closed source) | yes |
| File never sent to third party | yes | partial — cloud features | no — file is uploaded | yes |
| Auto-detect PII | yes | yes (paid tier) | limited | no |
| Real text deletion (not paint-over) | yes | yes | varies by tool | no — bytes remain |
| Free | yes | no ($20+/mo) | freemium with quotas | yes |
| Setup | git clone + 1 command | installer + account | none | none |
| Polish | hobby-grade | enterprise-grade | slick | none |

**When to use hush:** you're sharing a personal document with someone (lawyer, employer, doctor, recruiter), and the bits you're hiding genuinely matter.

**When hush isn't right:**
- You need legal-grade attestations for litigation or compliance — use a vendor with audit reports.
- You're processing thousands of documents in batch — there's no API, this is interactive.
- Your PDF is a scan with no text layer — run OCR first (`ocrmypdf`, Adobe, ABBYY).

---

## Architecture

```
              PDF (yours)
                  |
                  v
    +-------------+--------------+
    |  Browser tab               |
    |                            |
    |  WebGPU available?         |
    |                            |
    |     yes              no    |
    |      |               |     |
    +------|---------------|-----+
           v               v
   +---------------+   +---------------+
   | Browser       |   | Single hush   |
   | engine        |   | server        |
   |               |   |               |
   | mupdf-wasm    |   | pymupdf       |
   | Transformers  |   | transformers  |
   | .js + WebGPU  |   | + CPU         |
   |               |   |               |
   | runs in tab   |   | RAM only,     |
   |               |   | dropped after |
   |               |   | use or 15 min |
   +---------------+   +---------------+
           |                   |
           +---- neither -----+
                  |
            calls a cloud
            LLM API or a
            third party
                  |
                  v
            cleaned PDF
```

In the browser engine, the PDF bytes never cross a network boundary at all. In the server engine, they go to one server (the one you're connected to) and live in process memory only — not on disk, not in logs, not in any third-party service.

The detection model is downloaded once from Hugging Face's CDN on first run and cached locally. After that, no network traffic for the model either.

---

## Deploying it yourself

### Docker (recommended for VPS)

```bash
git clone https://github.com/samratjha96/hush-pdf
cd hush-pdf
docker compose up -d
```

The container listens on `0.0.0.0:8765`. Front it with a reverse proxy for TLS:

```caddyfile
hush.example.com {
  reverse_proxy localhost:8765
}
```

The first build downloads `openai/privacy-filter` into the image (~800 MB), so the image is ~2.5 GB but startup is instant.

To override how long sessions live in server memory:

```bash
HUSH_SESSION_TTL_SECONDS=300 docker compose up -d   # 5 minutes instead of 15
```

### Plain Python

```bash
uv run uvicorn app:app --host 127.0.0.1 --port 8765
```

Bind to `0.0.0.0` (or `::`) only if the host is intended to be reachable from outside.

---

## Troubleshooting

**"checking device" never resolves on the engine pill.**
The capability check is hung waiting for `navigator.gpu.requestAdapter()`. Reload the page. If it still hangs, your browser's WebGPU is unstable — click the engine pill (or run `localStorage.setItem('redactor.enginePref', 'server')` in DevTools and reload) to force the server engine.

**Model download is very slow on first run.**
Hugging Face Hub rate-limits anonymous downloads. Set a token before starting the server:

```bash
export HF_TOKEN=hf_xxx
uv run uvicorn app:app --host 127.0.0.1 --port 8765
```

**hush flagged a chunk that isn't actually PII.**
Uncheck it in the review pane. Only checked candidates are removed when you click Apply.

**hush missed something obvious.**
Click the engine pill to switch from "in this browser" to "local backend" — the server engine uses the full-precision model and recovers a few percent more entities than the q4-quantized browser model on dense documents. If it's still missed, drag a box around it manually.

**The cleaned PDF still shows the redacted text when I copy-paste.**
File a bug — this should not happen. There's a probe page at `/probe/` that verifies real-text-deletion against `assets/attention-is-all-you-need.pdf`. Run it and include the output in the bug report.

---

## Limitations

- **Scanned PDFs detect nothing.** hush operates on the text layer; image-only PDFs need OCR first (`ocrmypdf`, ABBYY, Adobe).
- **English-only.** `openai/privacy-filter` is trained on English. Other languages get poor recall.
- **Q4 quantization in the browser.** The browser engine runs a 4-bit quantized model so it fits in WebGPU memory; recall is roughly 5–10% lower than the server engine on dense academic-style documents. Toggle to the server engine if you suspect a miss.
- **Large PDFs are slow.** Designed for documents up to a few hundred pages. Multi-thousand-page PDFs work but aren't the target use case.
- **No batch / API mode.** Interactive review is the point. There's no headless redaction endpoint.
- **No legal attestations.** This is a personal project. If you need redaction for litigation discovery or regulated-industry compliance, use a tool with formal audits.

---

## FAQ

**Why "hush"?**
Short, suggests privacy without saying "private," and works as a verb — you can hush a document.

**Is the hosted version actually private?**
The hosted version's WebGPU path is identical to running locally — your PDF never leaves the tab. The fallback path (Safari, Firefox) does upload to the server, but the server keeps it in RAM only, deletes it after you download the cleaned copy or 15 minutes idle, and never forwards it to any third party (no S3, no LLM APIs, no logs of file content). If even that's too much trust, run hush yourself — `git clone` and `uv run` is one command.

**Does it work on Safari / Firefox?**
Yes. They don't ship WebGPU (or it's behind a flag), so hush automatically uses the server engine. The UX is the same.

**What categories does it detect?**
Person, address, email, phone, URL, date, account, secret. The model is `openai/privacy-filter`; see [its model card](https://huggingface.co/openai/privacy-filter) for details.

**Why not save redactions as comments or annotations?**
Annotations and comments don't remove the underlying text. hush rewrites the PDF using mupdf's redaction primitive, which strips the bytes for real.

**Can I trust it for HIPAA / GDPR / legal discovery?**
No, not without doing your own diligence. There are no audits, no SOC 2, no formal attestations. Read the code, test it on your own documents, and decide. For regulated work, run it yourself on infrastructure you control — don't use a hosted instance.

**Can I run it offline?**
Yes, after the first run. The model and mupdf-wasm get cached on first run; after that, no internet needed. The Docker image bakes the model in, so it's offline from day one.

**Why two engines instead of one?**
The browser engine is the headline — your file genuinely never leaves the tab. But WebGPU isn't universal yet (Safari, older Chrome, some Linux setups), and falling back to a cloud LLM API would defeat the entire point. So when WebGPU isn't available, hush falls back to a single self-hosted server with no third-party dependencies. Same no-cloud-LLM guarantee, slightly weaker physical-locality guarantee.

---

## License

hush-pdf is intended to be released under **AGPL-3.0-or-later**, matching [mupdf-wasm](https://www.npmjs.com/package/mupdf), which it depends on. (mupdf is dual-licensed — AGPL-3.0-or-later or commercial. AGPL is the only one compatible with redistribution as open source without a commercial mupdf license.)

A `LICENSE` file is not yet committed. If you fork or redistribute before one lands, treat the code as AGPL-3.0-or-later.

---

## Credits

- [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) — the PII detection model.
- [`mupdf` / `mupdf-wasm`](https://mupdf.readthedocs.io/) — text extraction and redaction primitives.
- [`Transformers.js`](https://github.com/huggingface/transformers.js) — running models on WebGPU.
- [`FastAPI`](https://fastapi.tiangolo.com/) and [`uv`](https://docs.astral.sh/uv/) — the Python side.
