# hush

Black out names, addresses, account numbers, and other personal info in your PDF — without sending it to a cloud AI service.

```
+--------+      +----------------+      +-----------+
| Drop a | ---> | Review what's  | ---> | Download  |
| PDF    |      | flagged        |      | clean PDF |
+--------+      +----------------+      +-----------+
```

## What it does

You drop a PDF. hush highlights everything it thinks is sensitive — names, emails, addresses, phone numbers, account numbers, dates. You uncheck anything it got wrong, draw boxes on anything it missed, then download a clean copy.

The cleaned copy has the underlying text **actually deleted**, not just covered with a black rectangle. Copy-paste from a redacted area produces nothing.

---

## Is my file safe?

Yes — specifically:

- **No third-party AI.** Detection runs on a small open-source model we host ourselves. Your document is never sent to OpenAI, Anthropic, Google, or any other AI service.
- **Memory only.** Your PDF is never written to disk, never logged, never backed up.
- **Deleted after use.** When you download the cleaned PDF, the original is dropped from server memory immediately. If you walk away, it's gone within 15 minutes either way.
- **Even better: it might not leave your device at all.** On Chrome and Edge, hush runs detection inside the browser tab itself using WebGPU — your file never gets uploaded.
- **Or run it yourself.** Don't trust the hosted version? [Run it on your own machine.](#self-hosting) The file genuinely never leaves your network.

---

## How does it compare?

| | hush | Adobe Acrobat | smallpdf / iLovePDF | Black rectangles in Preview |
|---|:-:|:-:|:-:|:-:|
| Free | yes | $20+/mo | freemium | yes |
| Auto-detects sensitive info | yes | yes (paid) | limited | no |
| Actually deletes the text | yes | yes | varies | **no — text remains** |
| No cloud AI calls | yes | unclear | unclear | yes |

**Use hush when** you're sharing a personal document with a lawyer, employer, doctor, or recruiter — and the bits you're hiding actually matter.

**Don't use it when** you need legal-grade attestations (use a vendor with audit reports), or your PDF is a scan with no text layer (run OCR first with [`ocrmypdf`](https://github.com/ocrmypdf/OCRmyPDF)).

---

## Self-hosting

### Run it on your laptop

```bash
git clone https://github.com/samratjha96/hush-pdf
cd hush-pdf
uv run uvicorn app:app --host 127.0.0.1 --port 8765
```

Then open http://127.0.0.1:8765. The first run downloads the detection model (~800 MB) and caches it locally; you need [`uv`](https://docs.astral.sh/uv/) installed.

### Run it on a VPS

```bash
git clone https://github.com/samratjha96/hush-pdf
cd hush-pdf
docker compose up -d
```

Container listens on `127.0.0.1:8765`. Front it with a reverse proxy for HTTPS, e.g. Caddy:

```caddyfile
hush.example.com {
  reverse_proxy localhost:8765
}
```

To shorten how long uploads stay in server memory:

```bash
HUSH_SESSION_TTL_SECONDS=300 docker compose up -d   # 5 minutes instead of 15
```

---

## Limitations

- Scanned PDFs detect nothing — run OCR first.
- English only.
- No audits, no SOC 2. For regulated work, run it yourself on infrastructure you control.

---

## FAQ

**Does it work on Safari / Firefox?**
Yes — they don't ship WebGPU, so hush uses the server-side engine. The experience is identical; your file briefly lives in the server's RAM (deleted after use), where on Chrome/Edge it never leaves the tab.

**What does it detect?**
Names, addresses, emails, phone numbers, URLs, dates, account numbers, secrets. Powered by [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter).

**Why "hush"?**
Short, suggests privacy without saying "private," and works as a verb.

**Can I run it offline?**
Yes. The Docker image bakes the model in; the local install caches it after the first run.

---

## License

hush-pdf is intended to be released under **AGPL-3.0-or-later**, matching [mupdf-wasm](https://www.npmjs.com/package/mupdf), which it depends on. A `LICENSE` file is not yet committed — until one lands, treat the code as AGPL-3.0-or-later.

## Credits

[`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) · [`mupdf-wasm`](https://mupdf.readthedocs.io/) · [`Transformers.js`](https://github.com/huggingface/transformers.js) · [`FastAPI`](https://fastapi.tiangolo.com/) · [`uv`](https://docs.astral.sh/uv/)
