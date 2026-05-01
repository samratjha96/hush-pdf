from __future__ import annotations

import asyncio
import io
import json
import os
import re
import secrets
import subprocess
import tempfile
import time
from collections import defaultdict
from contextlib import asynccontextmanager

# Allow MPS to fall back to CPU for ops it doesn't implement.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import fitz
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline

classifier = None
device_in_use: str = "cpu"
INFERENCE_BATCH_SIZE = 4
MODEL_MIN_SCORE = 0.75

SESSIONS: dict[str, dict] = {}
SESSION_TTL_SECONDS = int(os.environ.get("HUSH_SESSION_TTL_SECONDS", "900"))
SESSION_SWEEP_INTERVAL = 60

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(
    r"https?://[^\s)\]<>\"']+|\bwww\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s)\]<>\"']*)?"
)
PHONE_RE = re.compile(
    r"""(?x)
    (?<!\w)(?<!\d)
    (?:
        \+\d{1,3}[\s.-]\d{1,4}[\s.-]\d{2,4}[\s.-]\d{2,4}(?:[\s.-]\d{2,4})?
      | \(\d{3}\)\s?\d{3}[\s.-]\d{4}
      | \d{3}[.\-]\d{3}[.\-]\d{4}
      | \+\d{10,15}
    )
    (?!\d)(?!\w)
    """
)


def _pick_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, device_in_use
    device = _pick_device()
    print(f"Loading openai/privacy-filter on {device}...")
    try:
        classifier = pipeline(
            task="token-classification",
            model="openai/privacy-filter",
            aggregation_strategy="simple",
            device=device,
        )
        _ = classifier("warmup John Doe john@example.com")
        device_in_use = device
    except Exception as e:
        if device != "cpu":
            print(f"  {device} failed ({e}), falling back to cpu")
            classifier = pipeline(
                task="token-classification",
                model="openai/privacy-filter",
                aggregation_strategy="simple",
                device="cpu",
            )
            device_in_use = "cpu"
        else:
            raise
    print(f"Model ready on {device_in_use}.")

    async def _sweep_loop() -> None:
        while True:
            await asyncio.sleep(SESSION_SWEEP_INTERVAL)
            _gc_sessions()

    sweeper = asyncio.create_task(_sweep_loop())
    try:
        yield
    finally:
        sweeper.cancel()
        SESSIONS.clear()


def _empty_device_cache() -> None:
    if device_in_use == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device_in_use == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _classify_batched(texts: list[str]) -> list[list[dict]]:
    if not texts or classifier is None:
        return [[] for _ in texts]
    out: list[list[dict]] = []
    for i in range(0, len(texts), INFERENCE_BATCH_SIZE):
        chunk = texts[i : i + INFERENCE_BATCH_SIZE]
        try:
            r = classifier(chunk, batch_size=len(chunk))
        except Exception as e:
            print(f"  chunk {i}-{i+len(chunk)} failed on {device_in_use}: {e}; retrying per-item on cpu")
            r = []
            cpu_clf = pipeline(
                task="token-classification",
                model="openai/privacy-filter",
                aggregation_strategy="simple",
                device="cpu",
            )
            for t in chunk:
                r.append(cpu_clf(t))
        if r and isinstance(r, list) and r and isinstance(r[0], dict):
            r = [r]
        out.extend(r)
        _empty_device_cache()
    return out


def _gc_sessions() -> None:
    now = time.time()
    expired = [k for k, v in SESSIONS.items() if now - v["created_at"] > SESSION_TTL_SECONDS]
    for k in expired:
        SESSIONS.pop(k, None)


app = FastAPI(lifespan=lifespan)
if os.path.isdir("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")
if os.path.isdir("probe"):
    app.mount("/probe", StaticFiles(directory="probe", html=True), name="probe")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hush PDF — private PDF cleanup</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #fafaf9;
    --panel: #ffffff;
    --line: #e7e5e4;
    --line-strong: #d6d3d1;
    --ink: #18181b;
    --ink-soft: #3f3f46;
    --muted: #71717a;
    --muted-2: #a1a1aa;
    --accent: #18181b;
    --redact: #e11d48;
    --redact-soft: rgba(225, 29, 72, 0.10);
    --redact-ring: rgba(225, 29, 72, 0.55);
    --pulse: rgba(217, 119, 6, 0.95);
    --pulse-soft: rgba(217, 119, 6, 0.18);
    --kept: #71717a;
    --kept-soft: rgba(113, 113, 122, 0.06);
    --err-bg: #fef2f2;
    --err-fg: #991b1b;
    --shadow-soft: 0 1px 2px rgba(15, 15, 15, 0.04), 0 12px 28px -18px rgba(15, 15, 15, 0.18);
    --shadow-pop: 0 4px 12px -6px rgba(15, 15, 15, 0.18), 0 32px 60px -28px rgba(15, 15, 15, 0.30);
    --ease: cubic-bezier(0.16, 1, 0.3, 1);
    --ease-soft: cubic-bezier(0.4, 0, 0.2, 1);
  }
  * { box-sizing: border-box; }
  html, body { background: var(--bg); color: var(--ink); }
  body {
    margin: 0;
    font-family: 'Geist', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }
  .mono { font-family: 'Geist Mono', ui-monospace, SFMono-Regular, monospace; }
  ::selection { background: rgba(24, 24, 27, 0.12); }

  .topbar {
    position: sticky; top: 0; z-index: 30;
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 28px;
    background: rgba(250, 250, 249, 0.82);
    backdrop-filter: saturate(140%) blur(10px);
    -webkit-backdrop-filter: saturate(140%) blur(10px);
    border-bottom: 1px solid var(--line);
  }
  .topbar .brand {
    display: flex; align-items: center; gap: 10px;
    font-weight: 600; font-size: 14px; letter-spacing: -0.01em;
  }
  .topbar .brand .dot {
    width: 8px; height: 8px; border-radius: 999px;
    background: var(--ink);
    box-shadow: 0 0 0 4px rgba(24, 24, 27, 0.06);
  }
  .topbar .meta { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 10px; flex: 1 1 auto; min-width: 0; padding: 0 14px; }
  .topbar .meta .sep { padding: 0 8px; color: var(--muted-2); }
  .topbar .engine-slot { display: flex; align-items: center; }
  .topbar .right { display: flex; gap: 8px; align-items: center; }
  .engine-pill {
    display: inline-flex; align-items: center; gap: 6px;
    height: 22px; padding: 0 9px;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: var(--panel);
    color: var(--ink-soft);
    font-size: 11px; font-weight: 500;
    letter-spacing: 0.005em;
    cursor: default; user-select: none;
    transition: border-color 160ms var(--ease), background 160ms var(--ease);
  }
  .engine-pill[data-engine-id="browser"] { color: #047857; border-color: rgba(4, 120, 87, 0.25); background: rgba(4, 120, 87, 0.04); }
  .engine-pill[data-engine-id="server"]  { color: #b45309; border-color: rgba(180, 83, 9, 0.28); background: rgba(180, 83, 9, 0.04); }
  .engine-pill[data-engine-id="loading"] { color: var(--muted); }
  .engine-pill-dot {
    width: 6px; height: 6px; border-radius: 50%; background: currentColor;
    box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.04);
  }
  .engine-pill[data-engine-id="loading"] .engine-pill-dot { animation: enginePulse 1.4s ease-in-out infinite; }
  body.busy-detect .engine-pill { opacity: 0.55; cursor: not-allowed !important; }
  body.busy-detect .engine-pill .engine-pill-dot { animation: enginePulse 1.4s ease-in-out infinite; }
  @keyframes enginePulse { 0%,100% { opacity: 0.35; } 50% { opacity: 1; } }

  button, .btn {
    font: inherit; font-weight: 500;
    padding: 9px 14px;
    border-radius: 10px;
    border: 1px solid var(--line-strong);
    background: var(--panel);
    color: var(--ink);
    cursor: pointer;
    transition: transform 160ms var(--ease), box-shadow 160ms var(--ease), background 160ms var(--ease), border-color 160ms var(--ease);
  }
  button:hover, .btn:hover { border-color: #a8a29e; box-shadow: 0 1px 2px rgba(15,15,15,0.05); }
  button:active, .btn:active { transform: translateY(1px); }
  button:disabled { opacity: 0.45; cursor: not-allowed; transform: none; box-shadow: none; }
  button.primary {
    background: var(--accent); color: #fff; border-color: var(--accent);
  }
  button.primary:hover { background: #000; border-color: #000; box-shadow: 0 6px 20px -8px rgba(15,15,15,0.45); }
  button.danger {
    background: var(--redact); color: #fff; border-color: var(--redact);
  }
  button.danger:hover { background: #be123c; border-color: #be123c; box-shadow: 0 6px 20px -8px rgba(190,18,60,0.55); }
  button.ghost { background: transparent; border-color: transparent; color: var(--muted); }
  button.ghost:hover { background: rgba(24,24,27,0.04); border-color: transparent; color: var(--ink); }

  /* ==================== UPLOAD STAGE ==================== */
  .upload-shell {
    min-height: calc(100dvh - 60px);
    display: grid;
    grid-template-columns: 5fr 6fr;
    gap: 0;
  }
  @media (max-width: 900px) {
    .upload-shell { grid-template-columns: 1fr; min-height: auto; }
    .upload-shell .right-col { padding: 32px 24px 64px; }
    .upload-shell .left-col { padding: 56px 24px 24px; }
  }
  .upload-shell .left-col {
    padding: 96px 64px 64px;
    display: flex; flex-direction: column; justify-content: center;
  }
  .upload-shell .right-col {
    padding: 64px 64px 64px 0;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
  }
  .kicker {
    font-size: 11px; font-weight: 500; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 18px;
    display: inline-flex; align-items: center; gap: 8px;
  }
  .kicker::before {
    content: ""; display: inline-block; width: 18px; height: 1px; background: var(--muted-2);
  }
  h1 {
    font-size: 56px; line-height: 0.95; letter-spacing: -0.035em; font-weight: 600;
    margin: 0 0 22px;
  }
  @media (max-width: 900px) { h1 { font-size: 40px; } }
  .lead {
    color: var(--ink-soft); font-size: 16px; max-width: 46ch; margin: 0 0 36px;
  }
  .promises {
    list-style: none; padding: 0; margin: 0 0 28px; display: flex; flex-direction: column; gap: 10px;
  }
  .promises li {
    display: flex; align-items: flex-start; gap: 12px;
    color: var(--ink-soft); font-size: 13px;
  }
  .promises li .pmark {
    flex: none; margin-top: 4px;
    width: 14px; height: 14px; border-radius: 999px;
    background: var(--ink); position: relative;
  }
  .promises li .pmark::after {
    content: ""; position: absolute;
    left: 4px; top: 6px; width: 4px; height: 7px;
    border: solid #fff; border-width: 0 1.5px 1.5px 0;
    transform: rotate(45deg);
  }

  .drop-card {
    width: 100%; max-width: 480px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 28px;
    padding: 28px;
    box-shadow: var(--shadow-pop);
    transition: transform 220ms var(--ease), box-shadow 220ms var(--ease), border-color 220ms var(--ease);
    position: relative; overflow: hidden;
  }
  .drop-card::before {
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(120% 80% at 100% 0%, rgba(24,24,27,0.04), transparent 60%);
    pointer-events: none;
  }
  .drop-card.over { transform: translateY(-2px); border-color: var(--ink); }
  .drop-card .drop-zone {
    border: 0;
    border-radius: 18px;
    padding: 56px 20px;
    text-align: center;
    cursor: pointer;
    transition: background 160ms var(--ease);
    position: relative;
  }
  .drop-card .drop-zone:hover { background: transparent; }
  .drop-card.over .drop-zone { background: transparent; }
  .drop-card input[type=file] { display: none; }
  .drop-icon {
    width: 38px; height: 38px; margin: 0 auto 16px;
    border-radius: 12px;
    border: 1px solid var(--line);
    background: var(--bg);
    display: grid; place-items: center;
    color: var(--ink);
    transition: transform 240ms var(--ease);
  }
  .drop-card.over .drop-icon { transform: translateY(-3px); }
  .drop-title { font-weight: 500; color: var(--ink); margin-bottom: 4px; }
  .drop-sub { color: var(--muted); font-size: 12.5px; }
  .drop-meta {
    margin-top: 22px; padding-top: 16px;
    border-top: 1px solid var(--line);
    display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
    font-size: 11.5px; color: var(--muted);
  }
  .drop-meta b { display: block; color: var(--ink); font-weight: 500; font-size: 12.5px; margin-bottom: 2px; letter-spacing: -0.005em; }

  .upload-status { width: 100%; max-width: 480px; margin-top: 18px; }
  .status {
    padding: 14px 16px; border-radius: 12px;
    background: var(--panel); border: 1px solid var(--line);
    font-size: 13px; color: var(--ink-soft);
    box-shadow: var(--shadow-soft);
    display: flex; align-items: center; gap: 12px;
  }
  .status.error { background: var(--err-bg); border-color: #fecaca; color: var(--err-fg); }
  .spinner {
    width: 14px; height: 14px; flex: none;
    border-radius: 999px;
    border: 1.5px solid rgba(24,24,27,0.18);
    border-top-color: var(--ink);
    animation: spin 720ms linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .status .status-body { flex: 1; min-width: 0; }
  .status .status-line { font-size: 13px; color: var(--ink-soft); }
  .status .status-sub { font-size: 11.5px; color: var(--muted); margin-top: 4px; font-variant-numeric: tabular-nums; }
  .status .progress-track {
    height: 6px; border-radius: 999px; background: rgba(24,24,27,0.06);
    overflow: hidden; margin-top: 8px;
  }
  .status .progress-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #047857, #10b981);
    width: 0%; transition: width 280ms cubic-bezier(0.16, 1, 0.3, 1);
  }
  .status .progress-fill.indeterminate {
    width: 35%;
    background: linear-gradient(90deg, transparent, #10b981 50%, transparent);
    animation: progressSlide 1.2s ease-in-out infinite;
  }
  @keyframes progressSlide {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(285%); }
  }

  /* ==================== REVIEW STAGE ==================== */
  .review-shell {
    display: grid;
    grid-template-columns: minmax(360px, 460px) 1fr;
    gap: 0;
    min-height: calc(100dvh - 60px);
  }
  @media (max-width: 1100px) {
    .review-shell { grid-template-columns: 1fr; }
  }

  .entity-pane {
    border-right: 1px solid var(--line);
    background: var(--panel);
    display: flex; flex-direction: column;
    height: calc(100dvh - 60px);
    position: sticky; top: 60px;
  }
  @media (max-width: 1100px) {
    .entity-pane { height: auto; position: static; border-right: none; border-bottom: 1px solid var(--line); }
  }
  .entity-pane .pane-head {
    padding: 18px 22px;
    border-bottom: 1px solid var(--line);
    display: flex; flex-direction: column; gap: 10px;
  }
  .summary {
    display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
  }
  .summary .count {
    font-family: 'Geist Mono', monospace;
    font-size: 28px; font-weight: 500; letter-spacing: -0.02em;
    color: var(--ink);
  }
  .summary .count .of { color: var(--muted-2); font-size: 22px; }
  .summary .label { font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
  .pane-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .pane-actions .grow { flex: 1; min-width: 8px; }
  .toolbar-link {
    background: none; border: none; padding: 4px 0;
    color: var(--muted); font-size: 12px; cursor: pointer;
    transition: color 120ms var(--ease);
  }
  .toolbar-link:hover { color: var(--ink); }
  .toolbar-link + .toolbar-link::before {
    content: "·"; color: var(--muted-2); padding: 0 8px;
  }

  .entity-list {
    flex: 1; overflow-y: auto;
    padding: 8px 0 80px;
    scrollbar-width: thin;
  }
  .entity-list::-webkit-scrollbar { width: 8px; }
  .entity-list::-webkit-scrollbar-thumb { background: var(--line); border-radius: 999px; }

  .cat-block {
    border-bottom: 1px solid var(--line);
    padding: 4px 0;
  }
  .cat-block:last-child { border-bottom: none; }
  .cat-head {
    position: sticky; top: 0;
    background: rgba(255,255,255,0.94);
    backdrop-filter: blur(8px);
    z-index: 2;
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 22px 8px;
    cursor: pointer;
  }
  .cat-head:hover { background: rgba(245,245,244,0.94); }
  .cat-head .l { display: flex; align-items: center; gap: 10px; }
  .cat-pill {
    font-family: 'Geist Mono', monospace;
    font-size: 10px; font-weight: 500; letter-spacing: 0.08em;
    padding: 3px 8px; border-radius: 6px;
    background: var(--bg); border: 1px solid var(--line); color: var(--ink-soft);
    text-transform: lowercase;
  }
  .cat-count { font-size: 11px; color: var(--muted); }
  .cat-head .r { display: flex; align-items: center; gap: 10px; }
  .group-toggle {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; color: var(--muted);
    cursor: pointer; user-select: none;
    padding: 4px 8px; border-radius: 6px;
    transition: background 120ms var(--ease);
  }
  .group-toggle:hover { background: var(--bg); color: var(--ink); }
  .chev { transition: transform 200ms var(--ease); display: inline-block; color: var(--muted-2); }
  .cat-block.collapsed .chev { transform: rotate(-90deg); }
  .cat-block.collapsed .cat-body { display: none; }

  .cat-body { padding: 4px 0 12px; }
  .ent-row {
    --i: 0;
    padding: 10px 22px 10px 22px;
    border-top: 1px solid var(--line);
    cursor: pointer;
    transition: background 120ms var(--ease);
    animation: slideIn 380ms var(--ease) both;
    animation-delay: calc(var(--i) * 18ms);
  }
  @keyframes slideIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .ent-row:hover { background: rgba(245,245,244,0.6); }
  .ent-row.has-current { background: rgba(217,119,6,0.06); }
  .ent-row .ent-summary {
    display: grid; grid-template-columns: auto 1fr auto; gap: 12px; align-items: start;
  }
  .ent-row .text {
    font-family: 'Geist Mono', monospace; font-size: 13px; color: var(--ink);
    word-break: break-word;
    line-height: 1.4;
  }
  .ent-row .occ-count {
    font-size: 11px; color: var(--muted);
    font-family: 'Geist Mono', monospace;
    align-self: center;
  }
  .ent-cb-wrap {
    width: 18px; height: 18px;
    margin-top: 1px;
    position: relative;
    flex: none;
  }
  /* custom checkbox */
  .ent-cb-wrap input,
  .occ-cb-wrap input,
  .group-toggle input { position: absolute; opacity: 0; pointer-events: none; }
  .cb-box {
    width: 18px; height: 18px;
    border: 1.5px solid var(--line-strong);
    border-radius: 5px;
    background: #fff;
    display: grid; place-items: center;
    cursor: pointer;
    transition: background 120ms var(--ease), border-color 120ms var(--ease), transform 120ms var(--ease);
  }
  .cb-box:hover { border-color: var(--ink); }
  .cb-box .tick {
    width: 10px; height: 10px;
    color: #fff;
    opacity: 0; transform: scale(0.7);
    transition: opacity 140ms var(--ease), transform 140ms var(--ease);
  }
  input:checked + .cb-box {
    background: var(--redact);
    border-color: var(--redact);
  }
  input:checked + .cb-box .tick { opacity: 1; transform: scale(1); }
  input:indeterminate + .cb-box { background: var(--redact); border-color: var(--redact); }
  input:indeterminate + .cb-box .tick { opacity: 1; transform: scale(1); }
  input:focus-visible + .cb-box { outline: 2px solid var(--ink); outline-offset: 2px; }

  .occ-list { margin-top: 8px; padding-left: 30px; display: none; }
  .ent-row.expanded .occ-list { display: block; animation: fadeIn 220ms var(--ease) both; }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
  .occ-row {
    display: grid; grid-template-columns: auto 1fr auto; gap: 10px; align-items: start;
    padding: 8px 8px 8px 0;
    border-radius: 8px;
    transition: background 100ms var(--ease);
    cursor: pointer;
  }
  .occ-row:hover { background: rgba(217,119,6,0.05); }
  .occ-row.active { background: rgba(217,119,6,0.10); }
  .occ-cb-wrap { width: 16px; height: 16px; position: relative; flex: none; margin-top: 1px; }
  .occ-cb-wrap .cb-box { width: 16px; height: 16px; border-radius: 4px; }
  .occ-cb-wrap .cb-box .tick { width: 9px; height: 9px; }
  .occ-row .snippet {
    font-size: 12px; color: var(--ink-soft); line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .occ-row .snippet .ctx { color: var(--muted); }
  .occ-row .snippet .mark {
    background: rgba(225,29,72,0.12);
    color: var(--ink);
    padding: 1px 4px; border-radius: 4px;
    font-family: 'Geist Mono', monospace; font-size: 11.5px;
    border: 1px solid rgba(225,29,72,0.25);
  }
  .occ-row.unchecked .snippet .mark {
    background: rgba(113,113,122,0.08);
    border-color: rgba(113,113,122,0.18);
    color: var(--muted);
  }
  .occ-row .page-pill {
    font-family: 'Geist Mono', monospace;
    font-size: 10.5px; color: var(--muted);
    background: var(--bg); border: 1px solid var(--line);
    padding: 2px 7px; border-radius: 999px;
    align-self: center;
    white-space: nowrap;
  }

  /* ==================== PREVIEW PANE ==================== */
  .preview-pane {
    background: var(--bg);
    display: flex; flex-direction: column;
    min-width: 0;
  }
  .preview-head {
    position: sticky; top: 60px; z-index: 5;
    padding: 14px 28px;
    background: rgba(250,250,249,0.85);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--line);
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
  }
  .preview-head .file {
    display: flex; align-items: center; gap: 10px; min-width: 0;
  }
  .preview-head .file .name {
    font-size: 13px; color: var(--ink); font-weight: 500;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 280px;
  }
  .preview-head .file .stats { font-size: 11.5px; color: var(--muted); }
  .pager {
    display: flex; align-items: center; gap: 6px;
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 4px;
  }
  .pager button {
    border: none; background: transparent; padding: 6px 10px; border-radius: 6px;
    color: var(--ink-soft); cursor: pointer;
    transition: background 120ms var(--ease);
  }
  .pager button:hover { background: var(--bg); }
  .pager button:disabled { opacity: 0.35; cursor: not-allowed; }
  .pager .page-input {
    font-family: 'Geist Mono', monospace; font-size: 12.5px;
    width: 44px; text-align: center;
    background: transparent; border: none;
    color: var(--ink); padding: 4px 0;
    outline: none;
  }
  .pager .page-input:focus { background: var(--bg); border-radius: 4px; }
  .pager .total {
    font-family: 'Geist Mono', monospace; font-size: 12.5px;
    color: var(--muted); padding: 0 6px 0 0;
  }

  .preview-canvas-wrap {
    flex: 1;
    padding: 28px 28px 80px;
    display: flex; justify-content: center; align-items: flex-start;
  }
  .canvas {
    position: relative;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 6px;
    box-shadow: var(--shadow-pop);
    overflow: hidden;
    max-width: 100%;
    transition: opacity 220ms var(--ease);
  }
  .canvas.loading { opacity: 0.5; }
  .canvas.marking { cursor: crosshair; }
  .canvas.marking svg.overlay { pointer-events: auto; cursor: crosshair; }
  .canvas.marking svg.overlay rect.ov-redact,
  .canvas.marking svg.overlay rect.ov-kept { pointer-events: none; }
  .ov-ghost {
    fill: rgba(217, 119, 6, 0.15);
    stroke: rgba(217, 119, 6, 0.85);
    stroke-width: 1.2;
    stroke-dasharray: 3 2;
    pointer-events: none;
    rx: 1.4;
  }
  .ov-manual {
    fill: rgba(124, 58, 237, 0.10);
    stroke: rgba(124, 58, 237, 0.55);
    stroke-width: 0.9;
  }
  .ov-manual:hover { fill: rgba(124, 58, 237, 0.18); }
  .head-right { display: flex; align-items: center; gap: 10px; }
  .tool-btn {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 7px 11px; border-radius: 9px;
    border: 1px solid var(--line);
    background: var(--panel); color: var(--ink-soft);
    font: inherit; font-size: 12.5px; font-weight: 500;
    cursor: pointer;
    transition: background 120ms var(--ease), border-color 120ms var(--ease), color 120ms var(--ease);
  }
  .tool-btn:hover { border-color: var(--line-strong); color: var(--ink); }
  .tool-btn.active {
    background: var(--ink); color: #fff; border-color: var(--ink);
  }
  .tool-btn .kbd {
    font-family: 'Geist Mono', monospace; font-size: 10px;
    padding: 1px 4px; border-radius: 3px;
    background: rgba(24,24,27,0.06); color: var(--muted);
    margin-left: 2px;
  }
  .tool-btn.active .kbd { background: rgba(255,255,255,0.18); color: rgba(255,255,255,0.85); }
  .mark-banner {
    position: absolute; left: 50%; top: 14px; transform: translateX(-50%);
    background: rgba(24,24,27,0.92); color: #fff;
    padding: 6px 12px; border-radius: 999px;
    font-size: 11.5px; font-weight: 500;
    box-shadow: 0 6px 20px -6px rgba(0,0,0,0.4);
    pointer-events: none;
    z-index: 4;
    animation: fadeIn 200ms var(--ease) both;
  }
  .mark-banner .kbd { font-family: 'Geist Mono', monospace; padding: 1px 5px; border-radius: 3px; background: rgba(255,255,255,0.16); margin-left: 6px; font-size: 10px; }
  .apply-error {
    position: absolute; left: 50%; top: 14px; transform: translateX(-50%);
    max-width: 540px;
    background: var(--err-bg); color: var(--err-fg);
    border: 1px solid #fecaca;
    padding: 10px 16px; border-radius: 10px;
    font-size: 12.5px; font-weight: 500; line-height: 1.45;
    box-shadow: 0 8px 24px -10px rgba(190,18,60,0.25);
    z-index: 5;
    display: flex; align-items: flex-start; gap: 10px;
    animation: fadeIn 200ms var(--ease) both;
  }
  .apply-error .x {
    background: transparent; border: 0; color: var(--err-fg);
    font: inherit; cursor: pointer; padding: 0; line-height: 1;
    opacity: 0.7;
  }
  .apply-error .x:hover { opacity: 1; }
  .cat-pill.manual {
    background: rgba(124, 58, 237, 0.08);
    border-color: rgba(124, 58, 237, 0.30);
    color: rgb(91, 33, 182);
  }
  .delete-btn {
    background: none; border: none;
    width: 22px; height: 22px;
    border-radius: 6px;
    color: var(--muted-2);
    cursor: pointer;
    display: grid; place-items: center;
    opacity: 0; transition: opacity 120ms var(--ease), background 120ms var(--ease), color 120ms var(--ease);
    padding: 0;
  }
  .ent-row:hover .delete-btn { opacity: 1; }
  .delete-btn:hover { background: var(--err-bg); color: var(--err-fg); opacity: 1 !important; }
  /* Redaction preview mode: show what the actual blacked-out PDF will look like */
  .canvas.preview-mode .ov-redact,
  .canvas.preview-mode .ov-manual {
    fill: #000 !important;
    stroke: #000 !important;
    stroke-width: 0 !important;
    fill-opacity: 1 !important;
  }
  .canvas.preview-mode .ov-redact:hover,
  .canvas.preview-mode .ov-manual:hover { fill: #000 !important; }
  .canvas.preview-mode .ov-kept { opacity: 0; pointer-events: none; }
  .canvas.preview-mode .ov-current {
    animation: none !important;
  }
  .canvas img {
    display: block; max-width: 100%; height: auto;
    user-select: none; -webkit-user-drag: none;
  }
  .canvas svg.overlay {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    pointer-events: none;
  }
  .canvas svg.overlay rect {
    pointer-events: auto;
    cursor: pointer;
    transition: opacity 160ms var(--ease);
  }
  .ov-redact {
    fill: var(--redact-soft);
    stroke: var(--redact-ring);
    stroke-width: 0.8;
  }
  .ov-redact:hover { fill: rgba(225,29,72,0.18); }
  .ov-kept {
    fill: var(--kept-soft);
    stroke: rgba(113,113,122,0.35);
    stroke-width: 0.6;
    stroke-dasharray: 2 2;
  }
  .ov-kept:hover { fill: rgba(113,113,122,0.12); }
  .ov-current {
    fill: var(--pulse-soft) !important;
    stroke: var(--pulse) !important;
    stroke-width: 1.4 !important;
    animation: pulse 1.6s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }

  .empty {
    text-align: center; padding: 80px 20px; color: var(--muted); line-height: 1.6;
  }
  .empty kbd {
    font-family: 'Geist Mono', monospace; font-size: 11px;
    padding: 2px 6px; border-radius: 4px;
    background: var(--surface-2); border: 1px solid var(--border-strong);
    color: var(--ink); margin: 0 1px;
  }

  /* ==================== DONE STAGE ==================== */
  .done-shell {
    min-height: calc(100dvh - 60px);
    display: grid;
    place-items: center;
    padding: 60px 24px;
  }
  .done-card {
    max-width: 540px; width: 100%;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 24px;
    box-shadow: var(--shadow-pop);
    padding: 40px 36px;
  }
  .done-card .seal {
    width: 44px; height: 44px;
    border-radius: 14px;
    background: var(--ink);
    display: grid; place-items: center;
    color: #fff;
    margin-bottom: 22px;
  }
  .done-card h2 {
    font-size: 24px; letter-spacing: -0.02em; margin: 0 0 8px;
  }
  .done-card .sub { color: var(--muted); margin: 0 0 24px; font-size: 14px; }
  .done-summary {
    list-style: none; padding: 0; margin: 0 0 28px;
    border-top: 1px solid var(--line);
  }
  .done-summary li {
    display: flex; align-items: baseline; justify-content: space-between;
    padding: 10px 0; border-bottom: 1px solid var(--line);
    font-size: 13px;
  }
  .done-summary li .lbl {
    font-family: 'Geist Mono', monospace; font-size: 11px;
    text-transform: lowercase; color: var(--muted);
    letter-spacing: 0.04em;
  }
  .done-summary li .val {
    font-family: 'Geist Mono', monospace; color: var(--ink);
  }
  .done-actions { display: flex; gap: 10px; flex-wrap: wrap; }
  .download-btn {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 11px 18px; border-radius: 12px;
    background: var(--ink); color: #fff;
    text-decoration: none; font-weight: 500; font-size: 14px;
    transition: transform 160ms var(--ease), box-shadow 160ms var(--ease);
  }
  .download-btn:hover { transform: translateY(-1px); box-shadow: 0 8px 20px -8px rgba(15,15,15,0.4); }
  .download-btn:active { transform: translateY(0); }

  .hidden { display: none !important; }
  .sr { position: absolute !important; width: 1px; height: 1px; overflow: hidden; clip: rect(1px,1px,1px,1px); white-space: nowrap; }
</style>
</head>
<body>

<div class="topbar">
  <div class="brand"><span class="dot"></span> Hush PDF</div>
  <div class="meta" id="topmeta"></div>
  <div class="engine-slot">
    <span class="engine-pill" id="engine-pill" data-engine-id="loading" title="Where detection runs">
      <span class="engine-pill-dot" aria-hidden="true"></span>
      <span class="engine-pill-text">private mode</span>
    </span>
  </div>
  <div class="right" id="topactions"></div>
</div>

<!-- ==================== UPLOAD STAGE ==================== -->
<section id="stage-upload" class="upload-shell">
  <div class="left-col">
    <div class="kicker">private PDF cleanup</div>
    <h1>The PDF stays<br>on your laptop.</h1>
    <p class="lead">Find names, addresses, emails, phone numbers, account details, and other sensitive text before you share a PDF.</p>
    <ul class="promises">
      <li><span class="pmark"></span><div><b id="promise-1-title">Nothing leaves this machine.</b> <span id="promise-1-body">Your PDF is checked here, never in the cloud.</span></div></li>
      <li><span class="pmark"></span><div><b>Real removal.</b> Selected text is removed from the PDF, not just covered up.</div></li>
      <li><span class="pmark"></span><div><b>You're in control.</b> Review every find, keep what belongs, and remove the rest.</div></li>
    </ul>
  </div>
  <div class="right-col">
    <div class="drop-card" id="drop-card">
      <label class="drop-zone" for="file" id="drop-zone">
        <input type="file" id="file" accept="application/pdf">
        <div class="drop-icon" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3v4a1 1 0 0 0 1 1h4"/><path d="M17 21H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7l5 5v11a2 2 0 0 1-2 2z"/><path d="M12 17v-6"/><path d="M9 14l3-3 3 3"/></svg>
        </div>
        <div class="drop-title">Drop a PDF, or click to choose</div>
        <div class="drop-sub">Up to a few hundred pages</div>
      </label>
      <div class="drop-meta">
        <div><b>Privacy</b><span id="drop-meta-inference">stays on this device</span></div>
        <div><b>Looks for</b>names, addresses, emails, phone numbers, links, dates, accounts, secrets</div>
        <div><b>Review</b>everything found is selected first</div>
        <div><b>Result</b><span id="drop-meta-io">removes selected text from the PDF</span></div>
      </div>
    </div>
    <div id="upload-status" class="upload-status"></div>
  </div>
</section>

<!-- ==================== REVIEW STAGE ==================== -->
<section id="stage-review" class="review-shell hidden">
  <aside class="entity-pane">
    <div class="pane-head">
      <div class="summary">
        <div>
          <div class="label">selected for removal</div>
          <div class="count"><span id="checked-count">0</span><span class="of">/<span id="total-count">0</span></span></div>
        </div>
        <div>
          <div class="label" style="text-align:right">unique</div>
          <div class="count" style="font-size: 18px;"><span id="unique-count">0</span></div>
        </div>
      </div>
      <div class="pane-actions">
        <button class="toolbar-link" id="select-all">Select all</button>
        <button class="toolbar-link" id="select-none">Deselect all</button>
        <span class="grow"></span>
      </div>
    </div>
    <div class="entity-list" id="entities"></div>
  </aside>
  <main class="preview-pane">
    <div class="preview-head">
      <div class="file">
        <div>
          <div class="name" id="preview-name"></div>
          <div class="stats" id="preview-stats"></div>
        </div>
      </div>
      <div class="head-right">
        <button class="tool-btn" id="preview-toggle" title="Preview the cleaned PDF (P)">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>
          Preview<span class="kbd">P</span>
        </button>
        <button class="tool-btn" id="mark-toggle" title="Mark an area to remove (M)">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3h6v6H3z"/><path d="M15 15h6v6h-6z"/><path d="M9 9l6 6"/></svg>
          Mark area<span class="kbd">M</span>
        </button>
        <div class="pager">
          <button id="prev-page" aria-label="Previous page">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg>
          </button>
          <input class="page-input mono" id="page-input" inputmode="numeric" value="1">
          <span class="total mono">/ <span id="page-total">1</span></span>
          <button id="next-page" aria-label="Next page">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"/></svg>
          </button>
        </div>
      </div>
    </div>
    <div class="preview-canvas-wrap" style="position: relative;">
      <div id="mark-banner" class="mark-banner hidden">Drag to mark a region · <span class="kbd">Esc</span> to exit</div>
      <div id="apply-error" class="apply-error hidden" role="alert"></div>
      <div class="canvas" id="canvas">
        <img id="page-img" alt="" draggable="false">
        <svg class="overlay" id="overlay" preserveAspectRatio="none"></svg>
      </div>
    </div>
  </main>
</section>

<!-- ==================== DONE STAGE ==================== -->
<section id="stage-done" class="done-shell hidden">
  <div class="done-card">
    <div class="seal" aria-hidden="true">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
    </div>
    <h2 id="done-title">Cleaned.</h2>
    <p class="sub" id="done-sub">Selected text was removed from the PDF. The original never left this machine.</p>
    <ul class="done-summary" id="done-summary"></ul>
    <div class="done-actions">
      <a class="download-btn" id="download-link" download="">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        Download cleaned PDF
      </a>
      <button class="ghost" id="redact-another">Clean another</button>
    </div>
  </div>
</section>

<script type="module">
const $ = id => document.getElementById(id);
const TICK_SVG = '<svg class="tick" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';

// ============================================================
//  ENGINE ABSTRACTION
//  - server: classic POST /api/* roundtrip
//  - browser: mupdf-wasm + transformers.js, no upload (TODO)
// ============================================================

async function checkCaps() {
  const caps = {
    webgpu: false,
    maxBufferMB: 0,
    deviceMemoryGB: navigator.deviceMemory ?? null,
    hardwareConcurrency: navigator.hardwareConcurrency ?? null,
    crossOriginIsolated: !!self.crossOriginIsolated,
  };
  if ('gpu' in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        caps.webgpu = true;
        caps.maxBufferMB = Math.floor(adapter.limits.maxBufferSize / 1024 / 1024);
        if (adapter.info) {
          caps.gpuVendor = adapter.info.vendor;
          caps.gpuArch = adapter.info.architecture;
        }
      }
    } catch (e) {
      caps.webgpuError = e.message;
    }
  }
  return caps;
}

async function _formatErr(r) {
  let detail = '';
  try { detail = await r.text(); } catch { detail = r.statusText; }
  try { const j = JSON.parse(detail); if (j && j.detail) detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch {}
  return detail || ('HTTP ' + r.status);
}

const ServerEngine = {
  id: 'server',
  label: 'on this device',
  needsModelLoad: false,
  isReady: true,
  async preload() {},
  async detect(file, progressCb) {
    const emit = (u) => { if (progressCb) progressCb(typeof u === 'string' ? { kind: 'msg', text: u } : u); };
    emit({ kind: 'msg', text: 'checking privately on this device…' });
    const fd = new FormData();
    fd.append('pdf', file);
    const r = await fetch('/api/detect', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(await _formatErr(r));
    emit({ kind: 'msg', text: 'parsing response…' });
    return r.json();
  },
  async pageUrl(sessionId, n) {
    return '/api/page/' + sessionId + '/' + n + '.png?zoom=1.5';
  },
  async redact(sessionId, occIds) {
    const r = await fetch('/api/redact', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ session_id: sessionId, accepted_occurrence_ids: occIds }),
    });
    if (!r.ok) throw new Error(await _formatErr(r));
    return r.blob();
  },
  async mark(sessionId, page, rect) {
    const r = await fetch('/api/mark', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ session_id: sessionId, page, rect }),
    });
    if (!r.ok) throw new Error(await _formatErr(r));
    const { entity } = await r.json();
    return entity;
  },
  async deleteEntity(sessionId, entityId) {
    const r = await fetch('/api/entity/' + sessionId + '/' + entityId, { method: 'DELETE' });
    if (!r.ok) throw new Error(await _formatErr(r));
  },
};

// ===== BrowserEngine: mupdf-wasm + transformers.js =====

const MUPDF_URL = 'https://cdn.jsdelivr.net/npm/mupdf@1.27.0/dist/mupdf.js';
const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest';

let _mupdf = null;
let _classifier = null;
let _classifierLoading = null;
const _bSessions = new Map();
const MODEL_MIN_SCORE = 0.75;

async function _loadMupdf() {
  if (_mupdf) return _mupdf;
  _mupdf = await import(MUPDF_URL);
  return _mupdf;
}

async function _loadClassifier(progressCb) {
  if (_classifier) return _classifier;
  if (_classifierLoading) return _classifierLoading;
  _classifierLoading = (async () => {
    const tf = await import(TRANSFORMERS_URL);
    const tracker = {};
    const wrapped = data => {
      if (!progressCb) return;
      if (data && data.status === 'progress' && data.file) {
        tracker[data.file] = (data.loaded || 0) / (data.total || 1) * 100;
        const arr = Object.values(tracker);
        const avg = arr.reduce((s, v) => s + v, 0) / arr.length;
        progressCb({ phase: 'download', percent: avg, files: Object.keys(tracker).length });
      } else if (data && data.status === 'ready') {
        progressCb({ phase: 'ready', percent: 100 });
      }
    };
    const clf = await tf.pipeline('token-classification', 'openai/privacy-filter', {
      device: 'webgpu',
      dtype: 'q4',
      progress_callback: wrapped,
    });
    _classifier = clf;
    _classifierLoading = null;
    return clf;
  })();
  return _classifierLoading;
}

function _coalesce(ents, gap = 2) {
  if (!ents.length) return [];
  const sorted = [...ents].sort((a, b) => a.start - b.start);
  const out = [{ ...sorted[0] }];
  for (let i = 1; i < sorted.length; i++) {
    const e = sorted[i], last = out[out.length - 1];
    const lL = last.entity_group || last.entity;
    const eL = e.entity_group || e.entity;
    if (lL === eL && e.start - last.end <= gap) {
      last.end = Math.max(last.end, e.end);
      last.score = Math.min(last.score, e.score);
    } else {
      out.push({ ...e });
    }
  }
  return out;
}

function _extractTextWithBboxes(stext) {
  const chars = [];
  const bboxes = [];
  stext.walk({
    beginLine() {
      if (chars.length > 0) { chars.push('\n'); bboxes.push(null); }
    },
    onChar(ch, _origin, _font, _size, quad) {
      const xs = [quad[0], quad[2], quad[4], quad[6]];
      const ys = [quad[1], quad[3], quad[5], quad[7]];
      chars.push(ch);
      bboxes.push([Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]);
    },
    endLine() {},
  });
  return { text: chars.join(''), bboxes };
}

function _bboxUnion(bboxes) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const b of bboxes) {
    if (!b) continue;
    if (b[0] < x0) x0 = b[0];
    if (b[1] < y0) y0 = b[1];
    if (b[2] > x1) x1 = b[2];
    if (b[3] > y1) y1 = b[3];
  }
  if (x0 === Infinity) return null;
  return [x0, y0, x1, y1];
}

function _snippetLocal(text, lo, hi, pad = 80) {
  const start = Math.max(0, lo - pad);
  const end = Math.min(text.length, hi + pad);
  const norm = s => s.replace(/\s+/g, ' ');
  let before = norm(text.slice(start, lo)).replace(/^\s+/, '');
  let match = norm(text.slice(lo, hi)).trim();
  let after = norm(text.slice(hi, end)).replace(/\s+$/, '');
  if (start > 0) before = '…' + before;
  if (end < text.length) after = after + '…';
  return { before, match, after };
}

function _newSessionId() {
  const buf = new Uint8Array(12);
  crypto.getRandomValues(buf);
  return 'b_' + Array.from(buf, b => b.toString(16).padStart(2, '0')).join('');
}

const BrowserEngine = {
  id: 'browser',
  label: 'in this browser',
  needsModelLoad: true,
  isReady: false,

  async preload(progressCb) {
    await Promise.all([_loadMupdf(), _loadClassifier(progressCb)]);
    this.isReady = true;
  },

  async detect(file, progressCb) {
    const emit = (u) => { if (progressCb) progressCb(typeof u === 'string' ? { kind: 'msg', text: u } : u); };
    emit({ kind: 'msg', text: 'opening PDF…' });
    const mupdf = await _loadMupdf();
    if (!_classifier) {
      emit({ kind: 'msg', text: 'first run: getting the private checker ready. Saved on this device for next time.' });
      await _loadClassifier(p => {
        if (p.phase === 'download') {
          emit({ kind: 'download', percent: p.percent, files: p.files });
        } else if (p.phase === 'ready') {
          emit({ kind: 'msg', text: 'model ready, warming up GPU…' });
        }
      });
    }
    this.isReady = true;

    const buf = new Uint8Array(await file.arrayBuffer());
    const doc = mupdf.PDFDocument.openDocument(buf, 'application/pdf');
    const pageCount = doc.countPages();
    const pageDims = [];
    const entitiesByKey = new Map();
    let nextOccId = 0;

    for (let i = 0; i < pageCount; i++) {
      emit({ kind: 'page', i: i + 1, total: pageCount });
      const page = doc.loadPage(i);
      const bb = page.getBounds();
      pageDims.push({ w: bb[2] - bb[0], h: bb[3] - bb[1] });

      const stext = page.toStructuredText('preserve-whitespace');
      const { text, bboxes } = _extractTextWithBboxes(stext);
      if (!text.trim()) continue;

      const raw = await _classifier(text, { aggregation_strategy: 'simple' });
      const ents = _coalesce(raw, 2);

      for (const e of ents) {
        const label = (e.entity_group || e.entity || '').toUpperCase();
        if (!label) continue;
        if (Number(e.score ?? 1) < MODEL_MIN_SCORE) continue;
        const span = bboxes.slice(e.start, e.end);
        const rect = _bboxUnion(span);
        if (!rect) continue;
        const word = text.slice(e.start, e.end).trim();
        if (!word) continue;
        const key = label + '\x00' + word.toLowerCase();
        let rec = entitiesByKey.get(key);
        if (!rec) {
          rec = { label, text: word, occurrences: [] };
          entitiesByKey.set(key, rec);
        }
        rec.occurrences.push({
          id: nextOccId++,
          page: i + 1,
          rect,
          snippet: _snippetLocal(text, e.start, e.end),
        });
      }
    }

    const sorted = [...entitiesByKey.values()].sort(
      (a, b) =>
        a.label.localeCompare(b.label) ||
        b.occurrences.length - a.occurrences.length ||
        a.text.localeCompare(b.text)
    );
    const entities = [];
    const occLookup = {};
    sorted.forEach((rec, idx) => {
      const ent = { id: idx, label: rec.label, text: rec.text, occurrences: rec.occurrences };
      for (const occ of ent.occurrences) {
        occLookup[occ.id] = { entity_id: idx, page: occ.page, rect: occ.rect };
      }
      entities.push(ent);
    });

    const sessionId = _newSessionId();
    _bSessions.set(sessionId, {
      pdfBytes: buf,
      doc,
      entities,
      occLookup,
      pageDimensions: pageDims,
      pageCount,
    });

    return {
      session_id: sessionId,
      pages: pageCount,
      page_dimensions: pageDims,
      entities,
    };
  },

  async pageUrl(sessionId, n) {
    const mupdf = await _loadMupdf();
    const sess = _bSessions.get(sessionId);
    if (!sess) throw new Error('session expired');
    const page = sess.doc.loadPage(n - 1);
    const pix = page.toPixmap(mupdf.Matrix.scale(1.5, 1.5), mupdf.ColorSpace.DeviceRGB, false);
    const png = pix.asPNG();
    return URL.createObjectURL(new Blob([png], { type: 'image/png' }));
  },

  async redact(sessionId, occIds) {
    const sess = _bSessions.get(sessionId);
    if (!sess) throw new Error('session expired or unknown');
    const byPage = new Map();
    for (const id of occIds) {
      const rec = sess.occLookup[id];
      if (!rec) continue;
      const list = byPage.get(rec.page - 1) || [];
      list.push(rec.rect);
      byPage.set(rec.page - 1, list);
    }
    for (const [pageIdx, rects] of byPage.entries()) {
      const page = sess.doc.loadPage(pageIdx);
      for (const rect of rects) {
        const annot = page.createAnnotation('Redact');
        annot.setRect(rect);
      }
      page.applyRedactions();
    }
    const out = sess.doc.saveToBuffer('garbage,deflate').asUint8Array();
    _bSessions.delete(sessionId);
    return new Blob([out], { type: 'application/pdf' });
  },

  async mark(sessionId, page, rect) {
    const sess = _bSessions.get(sessionId);
    if (!sess) throw new Error('session expired or unknown');
    if (page < 1 || page > sess.pageCount) throw new Error('page out of range');
    const [a, b, c, d] = rect;
    const r = [Math.min(a, c), Math.min(b, d), Math.max(a, c), Math.max(b, d)];
    if ((r[2] - r[0]) < 2 || (r[3] - r[1]) < 2) throw new Error('rect too small');

    const p = sess.doc.loadPage(page - 1);
    const stext = p.toStructuredText('preserve-whitespace');
    const { text, bboxes } = _extractTextWithBboxes(stext);
    const inside = [];
    for (let i = 0; i < text.length; i++) {
      const bb = bboxes[i];
      if (!bb) { inside.push(' '); continue; }
      const cx = (bb[0] + bb[2]) / 2, cy = (bb[1] + bb[3]) / 2;
      if (cx >= r[0] && cx <= r[2] && cy >= r[1] && cy <= r[3]) inside.push(text[i]);
    }
    let underText = inside.join('').replace(/\s+/g, ' ').trim();
    const display = underText
      ? (underText.length <= 80 ? underText : underText.slice(0, 79).replace(/\s+$/, '') + '…')
      : '[marked region]';

    const nextOccId = (Object.keys(sess.occLookup).length === 0 ? 0 : Math.max(...Object.keys(sess.occLookup).map(Number)) + 1);
    const nextEntId = sess.entities.reduce((m, e) => Math.max(m, e.id), -1) + 1;
    const occ = { id: nextOccId, page, rect: r, snippet: { before: '', match: display, after: '' } };
    const ent = { id: nextEntId, label: 'MANUAL', text: display, occurrences: [occ] };
    sess.entities.push(ent);
    sess.occLookup[nextOccId] = { entity_id: nextEntId, page, rect: r };
    return ent;
  },

  async deleteEntity(sessionId, entityId) {
    const sess = _bSessions.get(sessionId);
    if (!sess) throw new Error('session expired or unknown');
    const ent = sess.entities.find(e => e.id === entityId);
    if (!ent) throw new Error('entity not found');
    if (ent.label !== 'MANUAL') throw new Error('only manual marks can be deleted');
    for (const occ of ent.occurrences) delete sess.occLookup[occ.id];
    sess.entities = sess.entities.filter(e => e.id !== entityId);
  },
};

function setEnginePill(engineId, label) {
  const pill = $('engine-pill');
  if (!pill) return;
  pill.dataset.engineId = engineId;
  pill.querySelector('.engine-pill-text').textContent = label;
  pill.title = engineTooltip(engineId);
}

function formatGpuLabel(fallback) {
  const c = window.__caps || {};
  if (!c.gpuVendor) return fallback;
  return c.gpuArch ? `${c.gpuVendor} ${c.gpuArch}` : c.gpuVendor;
}

function engineTooltip(engineId) {
  if (engineId === 'loading') return 'Checking this device…';
  if (engineId === 'browser') {
    const toggle = window.__browserCapable ? '\nClick to use the device app instead.' : '';
    return `Your PDF is checked in this browser tab and never leaves this device.${toggle}`;
  }
  if (engineId === 'server') {
    const toggle = window.__browserCapable ? '\nClick to check inside the browser tab instead.' : '';
    return `Your PDF is checked by the app running on this device. No cloud upload.${toggle}`;
  }
  return '';
}

function setPromiseCopy(engineId) {
  const title = $('promise-1-title');
  const body = $('promise-1-body');
  const dropInf = $('drop-meta-inference');
  const dropIo = $('drop-meta-io');
  if (engineId === 'browser') {
    if (title) title.textContent = 'Nothing leaves this browser tab.';
    if (body) body.textContent = 'Your PDF is checked in this tab, without a cloud upload.';
    if (dropInf) dropInf.textContent = 'stays in this browser tab';
    if (dropIo) dropIo.textContent = 'removes selected text from the PDF';
  } else {
    if (title) title.textContent = 'Nothing leaves this machine.';
    if (body) body.textContent = 'Your PDF is checked by the app on this device, never in the cloud.';
    if (dropInf) dropInf.textContent = 'stays on this device';
    if (dropIo) dropIo.textContent = 'removes selected text from the PDF';
  }
}

const ENGINE_MIN_BUFFER_MB = 1024;
const caps = await checkCaps();
window.__caps = caps;
const browserCapable = caps.webgpu && caps.maxBufferMB >= ENGINE_MIN_BUFFER_MB;

const ENGINE_PREF_KEY = 'redactor.enginePref';
function readEnginePref() {
  try { return localStorage.getItem(ENGINE_PREF_KEY); } catch { return null; }
}
function writeEnginePref(v) {
  try { localStorage.setItem(ENGINE_PREF_KEY, v); } catch {}
}

function pickEngine(forceId) {
  if (forceId === 'browser' && browserCapable) return BrowserEngine;
  return ServerEngine;
}

let engine = pickEngine();
setEnginePill(engine.id, engine.label);
setPromiseCopy(engine.id);
window.__engine = engine;
window.__BrowserEngine = BrowserEngine;
window.__ServerEngine = ServerEngine;
window.__browserCapable = browserCapable;

function switchEngine(id) {
  const next = id === 'browser' ? (browserCapable ? BrowserEngine : null) : ServerEngine;
  if (!next) return false;
  if (next.id === engine.id) return true;
  engine = next;
  window.__engine = engine;
  setEnginePill(engine.id, engine.label);
  setPromiseCopy(engine.id);
  writeEnginePref(engine.id);
  return true;
}
window.switchEngine = switchEngine;

// Click on the engine pill to toggle (only when both engines are usable).
let _detectInFlight = false;
const enginePillEl = $('engine-pill');
if (enginePillEl) {
  enginePillEl.style.cursor = browserCapable ? 'pointer' : 'help';
  enginePillEl.addEventListener('click', () => {
    if (!browserCapable) return;
    if (_detectInFlight) return;
    const review = $('stage-review');
    if (review && !review.classList.contains('hidden')) return; // mid-flow, ignore
    switchEngine(engine.id === 'browser' ? 'server' : 'browser');
  });
}

const stageUpload = $('stage-upload'), stageReview = $('stage-review'), stageDone = $('stage-done');
const dropCard = $('drop-card'), dropZone = $('drop-zone'), fileInput = $('file');
const uploadStatus = $('upload-status');
const entitiesEl = $('entities');
const pageImg = $('page-img'), overlayEl = $('overlay'), canvasEl = $('canvas');
const pageInput = $('page-input'), pageTotalEl = $('page-total');

const state = {
  session: null,
  fileName: '',
  checkedOcc: new Set(),       // occurrence ids that are marked for redaction
  occById: new Map(),           // occId -> {entityId, page, rect, snippet}
  entById: new Map(),           // entityId -> entity
  occByPage: new Map(),         // pageNum -> [occId, occId, ...]
  entByPage: new Map(),         // pageNum -> Set(entityId)
  currentPage: 1,
  currentOccId: null,           // hovered/focused occurrence
};
window.__state = state;

function pickFile(file) {
  if (!file) return;
  const isPdf = file.type === 'application/pdf' || /\.pdf$/i.test(file.name);
  if (!isPdf) {
    uploadStatus.innerHTML = '<div class="status error">' + escapeHtml(file.name) + ' is not a PDF.</div>';
    return;
  }
  detect(file);
}
dropCard.addEventListener('dragover', e => { e.preventDefault(); dropCard.classList.add('over'); });
dropCard.addEventListener('dragleave', () => dropCard.classList.remove('over'));
dropCard.addEventListener('drop', e => {
  e.preventDefault();
  dropCard.classList.remove('over');
  pickFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => pickFile(e.target.files[0]));
$('select-all').addEventListener('click', () => setAll(true));
$('select-none').addEventListener('click', () => setAll(false));
$('redact-another').addEventListener('click', reset);

$('prev-page').addEventListener('click', () => goToPage(state.currentPage - 1));
$('next-page').addEventListener('click', () => goToPage(state.currentPage + 1));
pageInput.addEventListener('change', e => {
  const n = parseInt(e.target.value, 10);
  if (!isNaN(n)) goToPage(n);
  else pageInput.value = state.currentPage;
});
document.addEventListener('keydown', e => {
  if (stageReview.classList.contains('hidden')) return;
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft') { goToPage(state.currentPage - 1); }
  else if (e.key === 'ArrowRight') { goToPage(state.currentPage + 1); }
});

function show(stage) {
  stageUpload.classList.toggle('hidden', stage !== 'upload');
  stageReview.classList.toggle('hidden', stage !== 'review');
  stageDone.classList.toggle('hidden', stage !== 'done');
  $('topactions').innerHTML = '';
  $('topmeta').innerHTML = '';
  if (stage === 'review') {
    $('topmeta').innerHTML = '<span>review</span><span class="sep">·</span><span class="mono">' + escapeHtml(state.fileName) + '</span>';
    const apply = document.createElement('button');
    apply.className = 'danger';
    apply.id = 'apply-btn';
    apply.textContent = 'Remove selected';
    apply.addEventListener('click', applyRedactions);
    const cancel = document.createElement('button');
    cancel.className = 'ghost';
    cancel.textContent = 'Start over';
    cancel.addEventListener('click', reset);
    $('topactions').appendChild(cancel);
    $('topactions').appendChild(apply);
  } else if (stage === 'done') {
    $('topmeta').innerHTML = '<span>done</span>';
  }
}

function reset() {
  state.session = null;
  state.fileName = '';
  state.checkedOcc.clear();
  state.occById.clear();
  state.entById.clear();
  state.occByPage.clear();
  state.entByPage.clear();
  state.currentPage = 1;
  state.currentOccId = null;
  fileInput.value = '';
  uploadStatus.innerHTML = '';
  entitiesEl.innerHTML = '';
  pageImg.removeAttribute('src');
  overlayEl.innerHTML = '';
  show('upload');
}

function renderStatusPanel({ lineHtml, sub, percent }) {
  const bar = percent == null
    ? '<div class="progress-fill indeterminate"></div>'
    : `<div class="progress-fill" style="width:${percent}%"></div>`;
  const subRow = sub ? `<div class="status-sub">${escapeHtml(sub)}</div>` : '';
  uploadStatus.innerHTML =
    `<div class="status">` +
      `<div class="spinner"></div>` +
      `<div class="status-body">` +
        `<div class="status-line">${lineHtml}</div>` +
        `<div class="progress-track">${bar}</div>` +
        subRow +
      `</div>` +
    `</div>`;
}

function renderUploadStatus(u, fname) {
  if (!u) { uploadStatus.innerHTML = ''; return; }
  if (u.kind === 'download') {
    const pct = Math.max(0, Math.min(100, Math.round(u.percent || 0)));
    const fileLabel = u.files ? `${u.files} file${u.files === 1 ? '' : 's'}` : '';
    renderStatusPanel({
      lineHtml: 'Getting the private checker ready · saved on this device for next time',
      sub: `${pct}% · ${fileLabel}`,
      percent: pct,
    });
    return;
  }
  if (u.kind === 'page') {
    const pct = u.total ? Math.round((u.i / u.total) * 100) : 0;
    renderStatusPanel({
      lineHtml: `Checking <span class="mono">${escapeHtml(fname || '')}</span>`,
      sub: `page ${u.i} / ${u.total}`,
      percent: pct,
    });
    return;
  }
  renderStatusPanel({ lineHtml: escapeHtml(u.text || '') });
}

async function detect(file) {
  state.fileName = file.name;
  const t0 = performance.now();
  _detectInFlight = true;
  document.body.classList.add('busy-detect');
  renderUploadStatus({ kind: 'msg', text: 'Checking ' + file.name + ' · this can take 30 seconds to a few minutes.' });
  try {
    state.session = await engine.detect(file, u => renderUploadStatus(u, file.name));
  } catch (err) {
    uploadStatus.innerHTML = '<div class="status error">' + escapeHtml(err.message) + '</div>';
    return;
  } finally {
    _detectInFlight = false;
    document.body.classList.remove('busy-detect');
  }
  const dt = ((performance.now() - t0) / 1000).toFixed(1);

  // Build indexes.
  state.occById.clear(); state.entById.clear();
  state.occByPage.clear(); state.entByPage.clear();
  state.checkedOcc.clear();
  for (const ent of state.session.entities) {
    state.entById.set(ent.id, ent);
    for (const occ of ent.occurrences) {
      state.occById.set(occ.id, { entityId: ent.id, ...occ });
      state.checkedOcc.add(occ.id);
      const p = occ.page;
      if (!state.occByPage.has(p)) state.occByPage.set(p, []);
      state.occByPage.get(p).push(occ.id);
      if (!state.entByPage.has(p)) state.entByPage.set(p, new Set());
      state.entByPage.get(p).add(ent.id);
    }
  }
  $('preview-name').textContent = file.name;
  $('preview-stats').textContent = state.session.pages + ' pages · analyzed in ' + dt + 's';
  pageTotalEl.textContent = state.session.pages;

  renderEntities();
  updateSummary();
  // Jump to first page that has an entity.
  const firstHit = state.session.entities.length > 0 ? state.session.entities[0].occurrences[0].page : 1;
  uploadStatus.innerHTML = '';
  show('review');
  goToPage(firstHit);
}

function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function catLabel(label) { return label.replace(/^PRIVATE_/, '').toLowerCase(); }

function renderEntities() {
  entitiesEl.innerHTML = '';
  if (!state.session.entities.length) {
    entitiesEl.innerHTML = '<div class="empty">Nothing detected automatically.<br>Press <kbd>M</kbd> on the page to mark areas yourself.</div>';
    return;
  }
  const byCat = {};
  for (const e of state.session.entities) (byCat[e.label] = byCat[e.label] || []).push(e);
  const cats = Object.keys(byCat).sort();
  let idx = 0;
  for (const cat of cats) {
    const ents = byCat[cat].sort((a, b) => b.occurrences.length - a.occurrences.length);
    const totalOcc = ents.reduce((s, e) => s + e.occurrences.length, 0);
    const block = document.createElement('div');
    block.className = 'cat-block';
    block.dataset.cat = cat;
    block.innerHTML = `
      <div class="cat-head">
        <div class="l">
          <span class="chev">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
          </span>
          <span class="cat-pill ${cat === 'MANUAL' ? 'manual' : ''}">${escapeHtml(catLabel(cat))}</span>
          <span class="cat-count">${ents.length} unique · ${totalOcc} occurrence${totalOcc === 1 ? '' : 's'}</span>
        </div>
        <div class="r">
          <label class="group-toggle" data-cat="${escapeHtml(cat)}">
            <input type="checkbox" class="group-cb" data-cat="${escapeHtml(cat)}">
            <span class="cb-box">${TICK_SVG}</span>
            <span>all</span>
          </label>
        </div>
      </div>
      <div class="cat-body"></div>
    `;
    const body = block.querySelector('.cat-body');
    for (const ent of ents) {
      const row = document.createElement('div');
      row.className = 'ent-row';
      row.style.setProperty('--i', idx++);
      row.dataset.entityId = ent.id;
      const occCount = ent.occurrences.length;
      const occList = ent.occurrences.map(occ => `
        <div class="occ-row" data-occ-id="${occ.id}" data-page="${occ.page}">
          <label class="occ-cb-wrap" onclick="event.stopPropagation()">
            <input type="checkbox" class="occ-cb" data-occ-id="${occ.id}">
            <span class="cb-box">${TICK_SVG}</span>
          </label>
          <div class="snippet">
            <span class="ctx">${escapeHtml(occ.snippet.before)}</span><span class="mark">${escapeHtml(occ.snippet.match)}</span><span class="ctx">${escapeHtml(occ.snippet.after)}</span>
          </div>
          <span class="page-pill">p.${occ.page}</span>
        </div>
      `).join('');
      const trailing = ent.label === 'MANUAL'
        ? `<button class="delete-btn" data-entity-id="${ent.id}" title="Remove this mark" onclick="event.stopPropagation()">
             <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/></svg>
           </button>`
        : `<span class="occ-count">×${occCount}</span>`;
      row.innerHTML = `
        <div class="ent-summary">
          <label class="ent-cb-wrap" onclick="event.stopPropagation()">
            <input type="checkbox" class="ent-cb" data-entity-id="${ent.id}">
            <span class="cb-box">${TICK_SVG}</span>
          </label>
          <div class="text">${escapeHtml(ent.text)}</div>
          ${trailing}
        </div>
        <div class="occ-list">${occList}</div>
      `;
      body.appendChild(row);
    }
    entitiesEl.appendChild(block);

    // Category collapse.
    block.querySelector('.cat-head').addEventListener('click', e => {
      if (e.target.closest('.group-toggle')) return;
      block.classList.toggle('collapsed');
    });
  }

  // Wire delete buttons (manual entities only).
  for (const btn of entitiesEl.querySelectorAll('.delete-btn')) {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const eid = parseInt(btn.dataset.entityId, 10);
      await deleteEntity(eid);
    });
  }

  // Wire entity expansion (click on .ent-summary anywhere except checkbox or delete).
  for (const row of entitiesEl.querySelectorAll('.ent-row')) {
    row.querySelector('.ent-summary').addEventListener('click', e => {
      if (e.target.closest('.ent-cb-wrap')) return;
      if (e.target.closest('.delete-btn')) return;
      row.classList.toggle('expanded');
      // Jump to first occurrence of this entity if expanding.
      if (row.classList.contains('expanded')) {
        const firstOcc = row.querySelector('.occ-row');
        if (firstOcc) {
          const p = parseInt(firstOcc.dataset.page, 10);
          if (p !== state.currentPage) goToPage(p);
        }
      }
    });
  }

  // Wire occurrence rows.
  for (const occRow of entitiesEl.querySelectorAll('.occ-row')) {
    const occId = parseInt(occRow.dataset.occId, 10);
    occRow.addEventListener('click', e => {
      if (e.target.closest('.occ-cb-wrap')) return;
      const p = parseInt(occRow.dataset.page, 10);
      state.currentOccId = occId;
      if (p !== state.currentPage) {
        goToPage(p);
      } else {
        renderOverlay();
      }
      highlightOccRow(occId);
    });
    occRow.addEventListener('mouseenter', () => {
      if (state.currentPage === parseInt(occRow.dataset.page, 10)) {
        state.currentOccId = occId;
        renderOverlay();
      }
    });
    occRow.addEventListener('mouseleave', () => {
      if (state.currentOccId === occId) {
        state.currentOccId = null;
        renderOverlay();
      }
    });
  }

  // Wire checkboxes.
  for (const cb of entitiesEl.querySelectorAll('.occ-cb')) {
    const id = parseInt(cb.dataset.occId, 10);
    cb.checked = state.checkedOcc.has(id);
    cb.addEventListener('change', () => {
      if (cb.checked) state.checkedOcc.add(id); else state.checkedOcc.delete(id);
      const occRow = cb.closest('.occ-row');
      if (occRow) occRow.classList.toggle('unchecked', !cb.checked);
      refreshEntityCheckbox(state.occById.get(id).entityId);
      refreshGroupCheckbox(state.entById.get(state.occById.get(id).entityId).label);
      updateSummary();
      renderOverlay();
    });
    const occRow = cb.closest('.occ-row');
    if (occRow) occRow.classList.toggle('unchecked', !cb.checked);
  }
  for (const cb of entitiesEl.querySelectorAll('.ent-cb')) {
    const entityId = parseInt(cb.dataset.entityId, 10);
    cb.addEventListener('change', () => {
      const ent = state.entById.get(entityId);
      for (const occ of ent.occurrences) {
        if (cb.checked) state.checkedOcc.add(occ.id); else state.checkedOcc.delete(occ.id);
        const occCb = entitiesEl.querySelector('.occ-cb[data-occ-id="' + occ.id + '"]');
        if (occCb) occCb.checked = cb.checked;
        const occRow = occCb && occCb.closest('.occ-row');
        if (occRow) occRow.classList.toggle('unchecked', !cb.checked);
      }
      refreshGroupCheckbox(ent.label);
      updateSummary();
      renderOverlay();
    });
    refreshEntityCheckbox(entityId);
  }
  for (const cb of entitiesEl.querySelectorAll('.group-cb')) {
    const cat = cb.dataset.cat;
    cb.addEventListener('change', () => {
      for (const ent of state.session.entities.filter(e => e.label === cat)) {
        for (const occ of ent.occurrences) {
          if (cb.checked) state.checkedOcc.add(occ.id); else state.checkedOcc.delete(occ.id);
          const occCb = entitiesEl.querySelector('.occ-cb[data-occ-id="' + occ.id + '"]');
          if (occCb) occCb.checked = cb.checked;
          const occRow = occCb && occCb.closest('.occ-row');
          if (occRow) occRow.classList.toggle('unchecked', !cb.checked);
        }
        const entCb = entitiesEl.querySelector('.ent-cb[data-entity-id="' + ent.id + '"]');
        if (entCb) { entCb.checked = cb.checked; entCb.indeterminate = false; }
      }
      updateSummary();
      renderOverlay();
    });
    refreshGroupCheckbox(cat);
  }
}

function refreshEntityCheckbox(entityId) {
  const ent = state.entById.get(entityId);
  if (!ent) return;
  const total = ent.occurrences.length;
  const on = ent.occurrences.filter(o => state.checkedOcc.has(o.id)).length;
  const cb = entitiesEl.querySelector('.ent-cb[data-entity-id="' + entityId + '"]');
  if (!cb) return;
  cb.checked = on > 0;
  cb.indeterminate = on > 0 && on < total;
}
function refreshGroupCheckbox(cat) {
  const ents = state.session.entities.filter(e => e.label === cat);
  const totalOcc = ents.reduce((s, e) => s + e.occurrences.length, 0);
  let on = 0;
  for (const e of ents) on += e.occurrences.filter(o => state.checkedOcc.has(o.id)).length;
  const cb = entitiesEl.querySelector('.group-cb[data-cat="' + cat + '"]');
  if (!cb) return;
  cb.checked = on > 0;
  cb.indeterminate = on > 0 && on < totalOcc;
}

function updateSummary() {
  const total = state.occById.size;
  $('total-count').textContent = total;
  $('checked-count').textContent = state.checkedOcc.size;
  $('unique-count').textContent = state.session ? state.session.entities.length : 0;
  const apply = $('apply-btn');
  if (apply) {
    const hasSel = state.checkedOcc.size > 0;
    apply.disabled = !hasSel;
    apply.title = hasSel ? '' : 'Select at least one item';
  }
}

function setAll(on) {
  state.checkedOcc.clear();
  if (on) for (const id of state.occById.keys()) state.checkedOcc.add(id);
  for (const cb of entitiesEl.querySelectorAll('.occ-cb')) {
    cb.checked = on;
    const occRow = cb.closest('.occ-row');
    if (occRow) occRow.classList.toggle('unchecked', !on);
  }
  for (const cb of entitiesEl.querySelectorAll('.ent-cb')) { cb.checked = on; cb.indeterminate = false; }
  for (const cb of entitiesEl.querySelectorAll('.group-cb')) { cb.checked = on; cb.indeterminate = false; }
  updateSummary();
  renderOverlay();
}

function highlightOccRow(occId) {
  for (const row of entitiesEl.querySelectorAll('.occ-row.active')) row.classList.remove('active');
  const target = entitiesEl.querySelector('.occ-row[data-occ-id="' + occId + '"]');
  if (target) {
    target.classList.add('active');
    target.scrollIntoView({ block: 'center', behavior: 'smooth' });
    // Mark parent ent-row as having current.
    for (const row of entitiesEl.querySelectorAll('.ent-row.has-current')) row.classList.remove('has-current');
    const entRow = target.closest('.ent-row');
    if (entRow) {
      entRow.classList.add('has-current', 'expanded');
    }
  }
}

function goToPage(n) {
  if (!state.session) return;
  n = Math.max(1, Math.min(state.session.pages, n));
  if (n === state.currentPage && pageImg.src) {
    pageInput.value = n;
    return;
  }
  state.currentPage = n;
  pageInput.value = n;
  $('prev-page').disabled = n <= 1;
  $('next-page').disabled = n >= state.session.pages;
  loadPage(n);
}

let _lastPageBlobUrl = null;
async function loadPage(n) {
  canvasEl.classList.add('loading');
  const dim = state.session.page_dimensions[n - 1];
  if (dim) {
    overlayEl.setAttribute('viewBox', '0 0 ' + dim.w + ' ' + dim.h);
  }
  let url;
  try {
    url = await engine.pageUrl(state.session.session_id, n);
  } catch (err) {
    canvasEl.classList.remove('loading');
    console.error('page render failed', err);
    return;
  }
  pageImg.onload = () => {
    canvasEl.classList.remove('loading');
    renderOverlay();
    if (_lastPageBlobUrl && _lastPageBlobUrl !== url) {
      URL.revokeObjectURL(_lastPageBlobUrl);
    }
    _lastPageBlobUrl = url.startsWith('blob:') ? url : null;
  };
  pageImg.onerror = () => { canvasEl.classList.remove('loading'); };
  pageImg.src = url;
}

function renderOverlay() {
  if (!state.session) return;
  overlayEl.innerHTML = '';
  const occIds = state.occByPage.get(state.currentPage) || [];
  for (const id of occIds) {
    const occ = state.occById.get(id);
    if (!occ) continue;
    const r = occ.rect;
    const w = r[2] - r[0], h = r[3] - r[1];
    const padX = Math.max(0.5, w * 0.04);
    const padY = Math.max(0.5, h * 0.10);
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', r[0] - padX);
    rect.setAttribute('y', r[1] - padY);
    rect.setAttribute('width', w + 2 * padX);
    rect.setAttribute('height', h + 2 * padY);
    rect.setAttribute('rx', '1.4');
    const checked = state.checkedOcc.has(id);
    const ent = state.entById.get(occ.entityId);
    const isManual = ent && ent.label === 'MANUAL';
    const baseClass = checked ? (isManual ? 'ov-manual' : 'ov-redact') : 'ov-kept';
    rect.setAttribute('class', baseClass + (state.currentOccId === id ? ' ov-current' : ''));
    rect.dataset.occId = id;
    rect.addEventListener('click', () => {
      const cb = entitiesEl.querySelector('.occ-cb[data-occ-id="' + id + '"]');
      if (cb) { cb.checked = !cb.checked; cb.dispatchEvent(new Event('change')); }
      highlightOccRow(id);
    });
    rect.addEventListener('mouseenter', () => {
      state.currentOccId = id;
      renderOverlay();
    });
    rect.addEventListener('mouseleave', () => {
      if (state.currentOccId === id) state.currentOccId = null;
      renderOverlay();
    });
    overlayEl.appendChild(rect);
  }
}

async function deleteEntity(entityId) {
  const ent = state.entById.get(entityId);
  if (!ent) return;
  try {
    await engine.deleteEntity(state.session.session_id, entityId);
  } catch (err) {
    console.warn('delete rejected:', err.message);
    return;
  }
  // Remove from indexes.
  for (const occ of ent.occurrences) {
    state.checkedOcc.delete(occ.id);
    state.occById.delete(occ.id);
    if (state.occByPage.has(occ.page)) {
      state.occByPage.set(occ.page, state.occByPage.get(occ.page).filter(id => id !== occ.id));
    }
  }
  state.entById.delete(entityId);
  state.session.entities = state.session.entities.filter(e => e.id !== entityId);
  renderEntities();
  updateSummary();
  renderOverlay();
}

// ==================== REDACTION PREVIEW TOGGLE ====================
const previewToggle = $('preview-toggle');
state.previewMode = false;
previewToggle.addEventListener('click', () => setPreviewMode(!state.previewMode));
function setPreviewMode(on) {
  state.previewMode = on;
  previewToggle.classList.toggle('active', on);
  canvasEl.classList.toggle('preview-mode', on);
}

// ==================== JUMP TO PREV/NEXT CANDIDATE PAGE ====================
function pagesWithOccurrences() {
  return Array.from(state.occByPage.keys())
    .filter(p => state.occByPage.get(p).length > 0)
    .sort((a, b) => a - b);
}
function jumpToCandidatePage(dir) {
  const hits = pagesWithOccurrences();
  if (!hits.length) return;
  if (dir > 0) {
    const next = hits.find(p => p > state.currentPage);
    if (next != null) goToPage(next); else goToPage(hits[0]);
  } else {
    const before = [...hits].reverse().find(p => p < state.currentPage);
    if (before != null) goToPage(before); else goToPage(hits[hits.length - 1]);
  }
}

// ==================== MANUAL MARK ====================
const markBanner = $('mark-banner');
const markToggle = $('mark-toggle');
state.markMode = false;
state.markDrag = null;

markToggle.addEventListener('click', () => setMarkMode(!state.markMode));
document.addEventListener('keydown', e => {
  if (stageReview.classList.contains('hidden')) return;
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'm' || e.key === 'M') { setMarkMode(!state.markMode); }
  else if (e.key === 'p' || e.key === 'P') { setPreviewMode(!state.previewMode); }
  else if (e.key === 'j' || e.key === 'J') { jumpToCandidatePage(1); }
  else if (e.key === 'k' || e.key === 'K') { jumpToCandidatePage(-1); }
  else if (e.key === 'Escape') {
    if (state.markMode) setMarkMode(false);
    else if (state.previewMode) setPreviewMode(false);
  }
});

function setMarkMode(on) {
  state.markMode = on;
  markToggle.classList.toggle('active', on);
  canvasEl.classList.toggle('marking', on);
  markBanner.classList.toggle('hidden', !on);
  if (!on && state.markDrag) cancelMarkDrag();
}

function clientToPdf(clientX, clientY) {
  const r = overlayEl.getBoundingClientRect();
  const dim = state.session.page_dimensions[state.currentPage - 1];
  if (!dim || r.width === 0 || r.height === 0) return [0, 0];
  const x = (clientX - r.left) / r.width * dim.w;
  const y = (clientY - r.top) / r.height * dim.h;
  return [x, y];
}

overlayEl.addEventListener('pointerdown', e => {
  if (!state.markMode) return;
  e.preventDefault();
  overlayEl.setPointerCapture(e.pointerId);
  const [x, y] = clientToPdf(e.clientX, e.clientY);
  const ghost = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  ghost.setAttribute('class', 'ov-ghost');
  ghost.setAttribute('x', x); ghost.setAttribute('y', y);
  ghost.setAttribute('width', 0); ghost.setAttribute('height', 0);
  overlayEl.appendChild(ghost);
  state.markDrag = { startX: x, startY: y, ghost, pointerId: e.pointerId };
});

overlayEl.addEventListener('pointermove', e => {
  if (!state.markDrag) return;
  const [x, y] = clientToPdf(e.clientX, e.clientY);
  const d = state.markDrag;
  d.curX = x; d.curY = y;
  const x0 = Math.min(d.startX, x), y0 = Math.min(d.startY, y);
  d.ghost.setAttribute('x', x0);
  d.ghost.setAttribute('y', y0);
  d.ghost.setAttribute('width', Math.abs(x - d.startX));
  d.ghost.setAttribute('height', Math.abs(y - d.startY));
});

overlayEl.addEventListener('pointerup', async e => {
  if (!state.markDrag) return;
  const d = state.markDrag;
  state.markDrag = null;
  try { overlayEl.releasePointerCapture(d.pointerId); } catch {}
  const x0 = Math.min(d.startX, d.curX || d.startX);
  const y0 = Math.min(d.startY, d.curY || d.startY);
  const x1 = Math.max(d.startX, d.curX || d.startX);
  const y1 = Math.max(d.startY, d.curY || d.startY);
  d.ghost.remove();
  if ((x1 - x0) < 3 || (y1 - y0) < 3) return; // ignore tiny drags
  await commitMark([x0, y0, x1, y1]);
});

overlayEl.addEventListener('pointercancel', cancelMarkDrag);

function cancelMarkDrag() {
  if (!state.markDrag) return;
  state.markDrag.ghost.remove();
  state.markDrag = null;
}

async function commitMark(rect) {
  let entity;
  try {
    entity = await engine.mark(state.session.session_id, state.currentPage, rect);
  } catch (err) {
    console.warn('mark rejected:', err.message);
    return;
  }
  // Append to session and indexes.
  state.session.entities.push(entity);
  state.entById.set(entity.id, entity);
  for (const occ of entity.occurrences) {
    state.occById.set(occ.id, { entityId: entity.id, ...occ });
    state.checkedOcc.add(occ.id);
    if (!state.occByPage.has(occ.page)) state.occByPage.set(occ.page, []);
    state.occByPage.get(occ.page).push(occ.id);
    if (!state.entByPage.has(occ.page)) state.entByPage.set(occ.page, new Set());
    state.entByPage.get(occ.page).add(entity.id);
  }
  renderEntities();
  updateSummary();
  renderOverlay();
  // Highlight the new entity row briefly.
  const newOccId = entity.occurrences[0].id;
  highlightOccRow(newOccId);
}

function showApplyError(msg) {
  const el = $('apply-error');
  el.classList.remove('hidden');
  el.innerHTML = '<div style="flex:1; min-width:0;"><div style="font-weight:600; margin-bottom:2px;">Could not remove selected text</div><div style="opacity:0.85;">' + escapeHtml(msg) + '</div></div><button class="x" type="button" aria-label="Dismiss">×</button>';
  el.querySelector('.x').addEventListener('click', () => el.classList.add('hidden'));
}

function resetApplyButton() {
  const btn = $('apply-btn');
  if (btn) { btn.disabled = false; btn.textContent = 'Remove selected'; }
}

async function applyRedactions() {
  const accepted = Array.from(state.checkedOcc);
  if (accepted.length === 0) {
    showApplyError('Select at least one item to remove.');
    return;
  }
  $('apply-error').classList.add('hidden');
  const btn = $('apply-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner" style="border-color: rgba(255,255,255,0.4); border-top-color: #fff; width: 12px; height: 12px;"></span>&nbsp;Applying';
  let blob;
  try {
    blob = await engine.redact(state.session.session_id, accepted);
  } catch (err) {
    showApplyError(err.message);
    resetApplyButton();
    return;
  }
  const url = URL.createObjectURL(blob);
  const counts = {};
  for (const id of accepted) {
    const occ = state.occById.get(id);
    if (!occ) continue;
    const ent = state.entById.get(occ.entityId);
    const lbl = ent ? ent.label : 'UNKNOWN';
    counts[lbl] = (counts[lbl] || 0) + 1;
  }
  const ul = $('done-summary');
  ul.innerHTML = '';
  const labels = Object.keys(counts).sort();
  if (!labels.length) {
    const li = document.createElement('li');
    li.innerHTML = '<span class="lbl">removed</span><span class="val">0</span>';
    ul.appendChild(li);
  }
  for (const k of labels) {
    const li = document.createElement('li');
    li.innerHTML = '<span class="lbl">' + escapeHtml(catLabel(k)) + '</span><span class="val">' + counts[k] + '</span>';
    ul.appendChild(li);
  }
  const totalLi = document.createElement('li');
  totalLi.innerHTML = '<span class="lbl"><b>total</b></span><span class="val"><b>' + accepted.length + '</b></span>';
  totalLi.style.borderBottom = 'none';
  ul.appendChild(totalLi);

  const dl = $('download-link');
  dl.href = url;
  dl.download = 'cleaned-' + state.fileName;
  const sub = $('done-sub');
  if (sub) {
    sub.textContent = 'Selected text was removed from the PDF. The original never left this machine.';
  }
  show('done');
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


def coalesce(entities: list[dict], gap_chars: int = 2) -> list[dict]:
    if not entities:
        return []
    es = sorted(entities, key=lambda e: e["start"])
    merged: list[dict] = [dict(es[0])]
    for e in es[1:]:
        last = merged[-1]
        last_label = last.get("entity_group") or last.get("entity")
        e_label = e.get("entity_group") or e.get("entity")
        if e_label == last_label and e["start"] - last["end"] <= gap_chars:
            last["end"] = max(last["end"], e["end"])
            last["score"] = min(last["score"], e["score"])
        else:
            merged.append(dict(e))
    return merged


def split_merged_span(label: str, text: str) -> list[str]:
    if label == "PRIVATE_EMAIL":
        return EMAIL_RE.findall(text) or [text]
    if label == "PRIVATE_URL":
        return URL_RE.findall(text) or [text]
    if label == "PRIVATE_PHONE":
        return PHONE_RE.findall(text) or [text]
    return [text]


def regex_backstop(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in EMAIL_RE.finditer(text):
        out.append(("PRIVATE_EMAIL", m.group()))
    for m in URL_RE.finditer(text):
        out.append(("PRIVATE_URL", m.group().rstrip(".,;:")))
    for m in PHONE_RE.finditer(text):
        s = m.group().strip()
        digits = sum(c.isdigit() for c in s)
        if digits >= 7:
            out.append(("PRIVATE_PHONE", s))
    return out


def entities_from_outputs(text: str, model_output: list[dict]) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    for e in coalesce(model_output):
        if float(e.get("score", 1.0)) < MODEL_MIN_SCORE:
            continue
        label = (e.get("entity_group") or e.get("entity") or "").upper()
        if not label:
            continue
        span = text[e["start"] : e["end"]].strip()
        for piece in split_merged_span(label, span):
            piece = piece.strip(" ,;:.()[]\"'")
            if len(piece) >= 3:
                seen.add((label, piece))
    for label, s in regex_backstop(text):
        s = s.strip(" ,;:.()[]\"'")
        if len(s) >= 3:
            seen.add((label, s))
    return seen


def liteparse_pdf(data: bytes) -> dict:
    in_path = ""
    out_path = ""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as in_f:
        in_f.write(data)
        in_path = in_f.name
    out_path = in_path + ".json"
    try:
        result = subprocess.run(
            [
                "npx", "-y", "@llamaindex/liteparse", "parse",
                in_path, "--format", "json", "--no-ocr", "-q", "-o", out_path,
            ],
            capture_output=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"liteparse failed: {result.stderr.decode()[:500]}",
            )
        with open(out_path) as f:
            return json.load(f)
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _snippet(text: str, lo: int, hi: int, pad: int = 80) -> dict:
    start = max(0, lo - pad)
    end = min(len(text), hi + pad)
    before = re.sub(r"\s+", " ", text[start:lo]).lstrip()
    match = re.sub(r"\s+", " ", text[lo:hi]).strip()
    after = re.sub(r"\s+", " ", text[hi:end]).rstrip()
    return {
        "before": ("…" + before) if start > 0 else before,
        "match": match,
        "after": (after + "…") if end < len(text) else after,
    }


def _find_all_offsets(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    out: list[int] = []
    nl = needle.lower()
    tl = text.lower()
    lo = 0
    while True:
        idx = tl.find(nl, lo)
        if idx < 0:
            break
        out.append(idx)
        lo = idx + max(1, len(nl))
    return out


def detect_pdf(data: bytes) -> dict:
    parsed = liteparse_pdf(data)
    doc = fitz.open(stream=data, filetype="pdf")

    page_texts = [p.get("text", "") for p in parsed.get("pages", [])]
    n = min(len(page_texts), doc.page_count)
    page_dimensions = [{"w": doc[i].rect.width, "h": doc[i].rect.height} for i in range(doc.page_count)]

    non_empty_indices = [i for i in range(n) if page_texts[i].strip()]
    non_empty_texts = [page_texts[i] for i in non_empty_indices]
    raw_outputs = _classify_batched(non_empty_texts)
    outputs_by_index = dict(zip(non_empty_indices, raw_outputs))

    by_key: dict[tuple[str, str], dict] = {}
    next_occ_id = 0

    for i in range(n):
        text = page_texts[i]
        if not text.strip():
            continue
        page = doc[i]
        for label, s in entities_from_outputs(text, outputs_by_index.get(i, [])):
            rects = page.search_for(s, quads=False)
            if not rects:
                continue
            offsets = _find_all_offsets(text, s)
            key = (label, s)
            rec = by_key.get(key)
            if rec is None:
                rec = {"label": label, "text": s, "occurrences": []}
                by_key[key] = rec
            for idx, r in enumerate(rects):
                if idx < len(offsets):
                    off = offsets[idx]
                elif offsets:
                    off = offsets[-1]
                else:
                    off = -1
                snip = _snippet(text, off, off + len(s)) if off >= 0 else {
                    "before": "", "match": s, "after": "",
                }
                rec["occurrences"].append({
                    "id": next_occ_id,
                    "page": i + 1,
                    "rect": [r.x0, r.y0, r.x1, r.y1],
                    "snippet": snip,
                })
                next_occ_id += 1

    page_count = doc.page_count
    doc.close()

    entities = []
    sorted_items = sorted(
        by_key.items(),
        key=lambda kv: (kv[0][0], -len(kv[1]["occurrences"]), kv[0][1]),
    )
    for ent_id, ((label, s), rec) in enumerate(sorted_items):
        entities.append({
            "id": ent_id,
            "label": label,
            "text": s,
            "occurrences": rec["occurrences"],
        })

    occurrences_by_id = {
        occ["id"]: {"entity_id": ent["id"], "page": occ["page"], "rect": occ["rect"]}
        for ent in entities
        for occ in ent["occurrences"]
    }

    session_id = secrets.token_urlsafe(16)
    SESSIONS[session_id] = {
        "pdf_bytes": data,
        "entities": entities,
        "pages": page_count,
        "page_dimensions": page_dimensions,
        "occurrences_by_id": occurrences_by_id,
        "created_at": time.time(),
    }
    _gc_sessions()
    return {
        "session_id": session_id,
        "pages": page_count,
        "page_dimensions": page_dimensions,
        "entities": entities,
    }


def apply_session_redactions(session_id: str, accepted_occ_ids: set[int]) -> tuple[bytes, dict[str, int]]:
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="session expired or unknown")

    occ_lookup = s["occurrences_by_id"]
    ent_by_id = {e["id"]: e for e in s["entities"]}

    by_page: dict[int, list[tuple[fitz.Rect, str]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for oid in accepted_occ_ids:
        rec = occ_lookup.get(oid)
        if not rec:
            continue
        ent = ent_by_id.get(rec["entity_id"])
        label = ent["label"] if ent else "UNKNOWN"
        by_page[rec["page"] - 1].append((fitz.Rect(*rec["rect"]), label))

    doc = fitz.open(stream=s["pdf_bytes"], filetype="pdf")
    for page_idx, items in by_page.items():
        page = doc[page_idx]
        for rect, label in items:
            page.add_redact_annot(rect, fill=(0, 0, 0))
            counts[label] += 1
        page.apply_redactions()

    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True, clean=True)
    doc.close()
    SESSIONS.pop(session_id, None)
    return buf.getvalue(), dict(counts)


@app.post("/api/detect")
async def api_detect(pdf: UploadFile = File(...)) -> JSONResponse:
    data = await pdf.read()
    return JSONResponse(detect_pdf(data))


class RedactRequest(BaseModel):
    session_id: str
    accepted_occurrence_ids: list[int]


@app.post("/api/redact")
async def api_redact(req: RedactRequest) -> Response:
    out_pdf, _ = apply_session_redactions(req.session_id, set(req.accepted_occurrence_ids))
    return Response(content=out_pdf, media_type="application/pdf")


@app.post("/api/discard/{session_id}")
def api_discard(session_id: str) -> JSONResponse:
    existed = SESSIONS.pop(session_id, None) is not None
    return JSONResponse({"discarded": existed})


class MarkRequest(BaseModel):
    session_id: str
    page: int
    rect: list[float]


@app.post("/api/mark")
def api_mark(req: MarkRequest) -> JSONResponse:
    s = SESSIONS.get(req.session_id)
    if not s:
        raise HTTPException(status_code=404, detail="session expired or unknown")
    if req.page < 1 or req.page > s["pages"]:
        raise HTTPException(status_code=404, detail="page out of range")
    if len(req.rect) != 4:
        raise HTTPException(status_code=400, detail="rect must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = req.rect
    rect = fitz.Rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    if rect.width < 2 or rect.height < 2:
        raise HTTPException(status_code=400, detail="rect too small")

    doc = fitz.open(stream=s["pdf_bytes"], filetype="pdf")
    try:
        page = doc[req.page - 1]
        rect = rect & page.rect
        if rect.is_empty:
            raise HTTPException(status_code=400, detail="rect is outside the page")
        words = page.get_text("words", clip=rect)
        text_under = re.sub(r"\s+", " ", " ".join(w[4] for w in words)).strip()
    finally:
        doc.close()

    if text_under:
        display_text = text_under if len(text_under) <= 80 else text_under[:79].rstrip() + "…"
    else:
        display_text = "[marked region]"
    next_ent_id = max((e["id"] for e in s["entities"]), default=-1) + 1
    next_occ_id = max(s["occurrences_by_id"].keys(), default=-1) + 1
    rect_coords = [rect.x0, rect.y0, rect.x1, rect.y1]
    occurrence = {
        "id": next_occ_id,
        "page": req.page,
        "rect": rect_coords,
        "snippet": {"before": "", "match": display_text, "after": ""},
    }
    entity = {
        "id": next_ent_id,
        "label": "MANUAL",
        "text": display_text,
        "occurrences": [occurrence],
    }
    s["entities"].append(entity)
    s["occurrences_by_id"][next_occ_id] = {
        "entity_id": next_ent_id,
        "page": req.page,
        "rect": rect_coords,
    }
    return JSONResponse({"entity": entity})


@app.delete("/api/entity/{session_id}/{entity_id}")
def api_delete_entity(session_id: str, entity_id: int) -> JSONResponse:
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="session expired or unknown")
    target = next((e for e in s["entities"] if e["id"] == entity_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="entity not found")
    if target["label"] != "MANUAL":
        raise HTTPException(status_code=400, detail="only manual marks can be deleted")
    for occ in target["occurrences"]:
        s["occurrences_by_id"].pop(occ["id"], None)
    s["entities"] = [e for e in s["entities"] if e["id"] != entity_id]
    return JSONResponse({"deleted": entity_id})


@app.get("/api/page/{session_id}/{page_num}.png")
def api_page(session_id: str, page_num: int, zoom: float = 1.5) -> Response:
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="session expired or unknown")
    if page_num < 1 or page_num > s["pages"]:
        raise HTTPException(status_code=404, detail="page out of range")
    zoom = max(0.5, min(3.0, zoom))
    doc = fitz.open(stream=s["pdf_bytes"], filetype="pdf")
    try:
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        png = pix.tobytes("png")
    finally:
        doc.close()
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=600"},
    )
