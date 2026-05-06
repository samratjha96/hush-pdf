# Hush PDF

Private PDF cleanup in your browser.

Hush PDF finds sensitive text in a PDF, lets you review every match, and removes the selected text from a cleaned copy. The PDF stays in the browser tab. There is no server upload and no cloud AI call.

## Why

Black rectangles are not privacy. If the text is still inside the PDF, copy-paste or PDF inspection can expose it later. Hush PDF uses real PDF redaction so selected text is removed from the file, not visually covered.

## Run Locally

Because the app is static, any local file server works:

```bash
python3 -m http.server 8765
```

Then open `http://127.0.0.1:8765`.

## Test

```bash
node tests/detection-rules.test.mjs
```

## Browser Requirements

Hush PDF needs WebGPU so the private checker can run locally in the browser.

- Use current Chrome or Microsoft Edge.
- Turn on hardware acceleration.
- Update graphics drivers if WebGPU is unavailable.
- Avoid Remote Desktop, VMs, or locked-down work browsers when possible.

Safari and Firefox are not supported until they ship the WebGPU support this app needs.

## Deploy

The Cloudflare deployment serves the static app from this repository:

```bash
npx wrangler deploy
```

## Credits

[`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) · [`mupdf-wasm`](https://mupdf.readthedocs.io/) · [`Transformers.js`](https://github.com/huggingface/transformers.js)
