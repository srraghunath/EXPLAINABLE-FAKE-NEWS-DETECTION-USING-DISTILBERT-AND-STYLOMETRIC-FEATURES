import os
import json
from urllib import error as url_error
from urllib import request as url_request
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request
from lime.lime_text import LimeTextExplainer
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
LABEL_NAMES = {0: "Real / likely true", 1: "Fake / likely false"}
LABEL_DESCRIPTIONS = {
    0: "The model associates this text with the real/true class.",
    1: "The model associates this text with the fake/false class.",
}
LIME_SAMPLES = int(os.getenv("LIME_SAMPLES", "300"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_KEY_FILE = ROOT / "groq_api_key.txt"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_KEY_FILE = ROOT / "openrouter_api_key.txt"

app = Flask(__name__)
models = {}

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VeriLens — Fake News XAI</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, Segoe UI, sans-serif; --ink: #edf2ff; --muted: #9aa8ca; --line: #29385e; --panel: rgba(18, 27, 55, .82); --accent: #73a7ff; --mint: #72e0ba; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; color: var(--ink); background: radial-gradient(circle at 15% 0%, #1b2e61 0, transparent 38%), radial-gradient(circle at 100% 20%, #123e4b 0, transparent 32%), #080e1d; }
    .shell { max-width: 1240px; margin: auto; padding: 28px 22px 70px; }
    .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 58px; }
    .brand { display: flex; align-items: center; gap: 10px; font-weight: 900; letter-spacing: -.03em; }
    .brand-mark { display: grid; place-items: center; width: 34px; height: 34px; border-radius: 11px; color: #07101f; background: linear-gradient(135deg, #9dc0ff, #72e0ba); box-shadow: 0 8px 20px #73a7ff35; }
    .nav-note { color: var(--muted); font-size: .82rem; }
    .hero { display: flex; justify-content: space-between; gap: 24px; align-items: end; margin-bottom: 34px; }
    .eyebrow { color: var(--mint); font-weight: 800; letter-spacing: .14em; text-transform: uppercase; font-size: .76rem; }
    h1 { margin: 8px 0 12px; font-size: clamp(2.4rem, 6vw, 4.8rem); line-height: .96; letter-spacing: -0.06em; }
    .subtitle { color: var(--muted); max-width: 720px; line-height: 1.65; margin: 0; }
    .status { border: 1px solid #356158; background: #132d2b; border-radius: 999px; padding: 10px 15px; color: #9ff0c0; white-space: nowrap; font-size: .88rem; }
    .card { background: var(--panel); backdrop-filter: blur(16px); border: 1px solid var(--line); border-radius: 24px; padding: 26px; box-shadow: 0 24px 80px #0005; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(340px, .95fr); gap: 22px; }
    .metrics { display: flex; gap: 10px; margin-top: 24px; flex-wrap: wrap; }
    .metric { padding: 9px 13px; border: 1px solid #29385e; border-radius: 12px; color: var(--muted); font-size: .78rem; background: #101a35aa; }
    .metric strong { color: var(--ink); margin-right: 4px; }
    label { display: block; font-size: .82rem; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; margin-bottom: 9px; color: #b7c1df; }
    select, textarea { width: 100%; color: var(--ink); background: #0b142b; border: 1px solid #334267; border-radius: 14px; padding: 14px 15px; font: inherit; outline: none; }
    select:focus, textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px #73a7ff22; }
    select { margin-bottom: 20px; }
    textarea { min-height: 280px; resize: vertical; line-height: 1.6; }
    .field-head { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
    .counter { color: var(--muted); font-size: .78rem; }
    .model-note { color: var(--muted); font-size: .8rem; margin: -10px 0 18px; line-height: 1.4; }
    .actions { display: flex; gap: 12px; align-items: center; margin-top: 18px; flex-wrap: wrap; }
    .secondary { color: #dce5ff; background: #1a284a; border: 1px solid #385181; box-shadow: none; }
    button { border: 0; border-radius: 14px; padding: 14px 22px; font-weight: 800; cursor: pointer; color: #07101f; background: linear-gradient(135deg, #9dc0ff, #72e0ba); box-shadow: 0 10px 25px #73a7ff30; }
    button:disabled { opacity: .6; cursor: wait; }
    .hint, .meta { color: var(--muted); font-size: .88rem; }
    .result { margin-top: 22px; display: none; }
    .result.show { display: block; }
    .result.error { border-color: #9f4657; }
    .prediction { display: flex; justify-content: space-between; align-items: center; gap: 16px; }
    .label-value { font-size: clamp(1.7rem, 4vw, 2.7rem); font-weight: 900; margin: 5px 0; }
    .confidence { font-size: 2rem; color: var(--mint); font-weight: 900; }
    .bar { height: 10px; background: #273353; border-radius: 99px; overflow: hidden; margin-top: 16px; }
    .bar span { display: block; height: 100%; background: linear-gradient(90deg, #73a7ff, #72e0ba); transition: width .5s ease; }
    .meta { margin-top: 12px; }
    .section-title { margin: 0 0 6px; font-size: 1.15rem; }
    .section-copy { margin: 0 0 18px; color: var(--muted); line-height: 1.5; font-size: .9rem; }
    .explanation { margin-top: 22px; }
    .feature { display: flex; align-items: center; gap: 10px; margin: 10px 0; }
    .feature-word { width: 110px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 700; }
    .feature-track { flex: 1; height: 9px; background: #202e50; border-radius: 99px; overflow: hidden; }
    .feature-fill { height: 100%; border-radius: inherit; }
    .feature-fill.support { background: var(--mint); }
    .feature-fill.against { background: #ff8296; }
    .feature-score { width: 58px; text-align: right; color: var(--muted); font-size: .8rem; }
    .llm { border-left: 3px solid var(--accent); padding: 12px 14px; background: #101c39; color: #dce5ff; line-height: 1.55; border-radius: 0 12px 12px 0; white-space: pre-wrap; }
    .legend { display: flex; gap: 16px; color: var(--muted); font-size: .78rem; margin-top: 14px; }
    .legend span::before { content: ""; display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; background: var(--mint); }
    .legend span:last-child::before { background: #ff8296; }
    .howto { margin-top: 22px; }
    .howto-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 18px; }
    .step { border: 1px solid #29385e; border-radius: 16px; padding: 17px; background: #0e1831aa; }
    .step-number { display: grid; place-items: center; width: 28px; height: 28px; border-radius: 9px; background: #243b70; color: #a9c5ff; font-weight: 900; margin-bottom: 12px; }
    .step h3 { margin: 0 0 6px; font-size: .96rem; }
    .step p { margin: 0; color: var(--muted); line-height: 1.45; font-size: .82rem; }
    .trust { display: flex; gap: 10px; align-items: flex-start; margin-top: 16px; color: var(--muted); font-size: .8rem; line-height: 1.5; }
    .trust strong { color: #f1d58a; }
    @media (max-width: 820px) { .topbar { margin-bottom: 38px; } .hero, .grid { display: block; } .status { display: inline-block; margin-top: 18px; } .grid > .card + .card { margin-top: 22px; } .howto-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark">V</span><span>VeriLens</span></div>
      <div class="nav-note">Local inference · transparent evidence · no fact-checking claim</div>
    </header>
    <section class="hero">
      <div>
        <div class="eyebrow">VeriLens / Explainable AI</div>
        <h1>Read the signal.<br>See the reason.</h1>
        <p class="subtitle">Analyze an article with a local DistilBERT classifier and inspect the words that moved its decision. This is a model assessment, not a substitute for fact-checking.</p>
      </div>
      <div class="status">● Local model + LIME</div>
    </section>
    <div class="metrics">
      <div class="metric"><strong>2</strong> classifier options</div>
      <div class="metric"><strong>10</strong> LIME signals shown</div>
      <div class="metric"><strong>128</strong> token analysis window</div>
    </div>
    <section class="grid">
      <section class="card">
        <form id="form">
          <label for="model">Model to inspect</label>
          <select id="model">
            <option value="baseline">Baseline model</option>
            <option value="robust">Robust model</option>
          </select>
          <div id="model-note" class="model-note">Fast local classifier trained on the source domain.</div>
          <div class="field-head"><label for="text">Article text</label><span id="counter" class="counter">0 characters</span></div>
          <textarea id="text" placeholder="Paste the headline or full article text here..." required></textarea>
          <div class="actions">
            <button id="submit" type="submit">Analyze article</button>
            <button id="sample" class="secondary" type="button">Try a sample</button>
            <span class="hint">LIME samples are generated for this article.</span>
          </div>
        </form>
        <div class="trust"><span>◆</span><span><strong>Use responsibly:</strong> this is a learned signal about writing patterns, not proof that a claim is true or false. Verify important claims with trusted sources.</span></div>
      </section>
      <section id="result" class="card result">
        <div class="prediction">
          <div><div class="hint">Model verdict</div><div id="prediction" class="label-value"></div><div id="description" class="hint"></div></div>
          <div id="confidence" class="confidence"></div>
        </div>
        <div class="bar"><span id="bar"></span></div>
        <div id="meta" class="meta"></div>
        <div class="explanation">
          <h2 class="section-title">Why did the model decide this?</h2>
          <p class="section-copy">LIME perturbs the article and estimates each word’s local influence. Green supports the verdict; pink pushes against it.</p>
          <div id="features"></div>
          <div class="legend"><span>Supports verdict</span><span>Pushes against</span></div>
        </div>
        <div class="explanation">
          <h2 class="section-title">AI explanation</h2>
          <p class="section-copy">The configured language model turns the LIME evidence into a readable summary. It does not independently verify the story.</p>
          <div id="llm" class="llm">Waiting for analysis…</div>
        </div>
      </section>
    </section>
    <section class="card howto">
      <h2 class="section-title">How to use VeriLens</h2>
      <p class="section-copy">Get a useful explanation in three simple steps.</p>
      <div class="howto-grid">
        <div class="step"><div class="step-number">1</div><h3>Paste the story</h3><p>Use a headline, a claim, or the full article text. More context gives the model more language to inspect.</p></div>
        <div class="step"><div class="step-number">2</div><h3>Choose a model</h3><p>Baseline is the fast reference model. Robust includes target-domain adaptation for a second perspective.</p></div>
        <div class="step"><div class="step-number">3</div><h3>Read the evidence</h3><p>Start with the verdict, then inspect LIME words and the AI summary. Green and pink show local influence.</p></div>
      </div>
    </section>
  </main>
  <script>
    const form = document.getElementById("form");
    const button = document.getElementById("submit");
    const result = document.getElementById("result");
    const textInput = document.getElementById("text");
    const counter = document.getElementById("counter");
    const model = document.getElementById("model");
    const modelNote = document.getElementById("model-note");
    const samples = {
      baseline: "The central bank published its economic outlook after the policy meeting, including projected inflation and investment figures.",
      robust: "Officials announced a public infrastructure program and released a detailed report describing its funding, timeline, and expected economic impact."
    };
    const modelNotes = {
      baseline: "Fast local classifier trained on the source domain.",
      robust: "Domain-adapted classifier that combines source data with target-domain examples."
    };
    const updateEditor = () => {
      counter.textContent = `${textInput.value.length.toLocaleString()} characters`;
      modelNote.textContent = modelNotes[model.value];
    };
    textInput.addEventListener("input", updateEditor);
    model.addEventListener("change", updateEditor);
    document.getElementById("sample").addEventListener("click", () => {
      textInput.value = samples[model.value];
      updateEditor();
      textInput.focus();
    });
    updateEditor();
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      button.textContent = "Analyzing...";
      result.className = "card result";
      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({model: document.getElementById("model").value, text: document.getElementById("text").value})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Prediction failed.");
        document.getElementById("prediction").textContent = `${data.label_name} (label ${data.label})`;
        document.getElementById("description").textContent = data.label_description;
        document.getElementById("confidence").textContent = `${data.confidence.toFixed(1)}%`;
        document.getElementById("bar").style.width = `${data.confidence}%`;
        document.getElementById("meta").textContent = `${data.model} • ${data.device} • ${data.characters} characters analyzed`;
        const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
        document.getElementById("features").innerHTML = data.lime.map((item) => {
          const width = Math.min(100, Math.max(8, Math.abs(item.weight) * 100));
          const direction = item.weight >= 0 ? "support" : "against";
          return `<div class="feature"><span class="feature-word" title="${escapeHtml(item.word)}">${escapeHtml(item.word)}</span><span class="feature-track"><span class="feature-fill ${direction}" style="width:${width}%"></span></span><span class="feature-score">${item.weight >= 0 ? "+" : ""}${item.weight.toFixed(3)}</span></div>`;
        }).join("");
        document.getElementById("llm").textContent = data.llm_explanation;
        result.className = "card result show";
      } catch (error) {
        document.getElementById("prediction").textContent = error.message;
        document.getElementById("confidence").textContent = "";
        document.getElementById("meta").textContent = "Check that the selected model exists and the article is not empty.";
        document.getElementById("features").innerHTML = "";
        document.getElementById("llm").textContent = "No explanation available.";
        result.className = "card result show error";
      } finally {
        button.disabled = false;
        button.textContent = "Analyze article";
      }
    });
  </script>
</body>
</html>
"""


def load_model(name):
    if name not in {"baseline", "robust"}:
        raise ValueError("Model must be 'baseline' or 'robust'.")
    if name not in models:
        model_path = ROOT / f"{name}_model"
        if not model_path.is_dir():
            raise FileNotFoundError(f"{model_path.name} is missing. Train that model first.")
        tokenizer = DistilBertTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = DistilBertForSequenceClassification.from_pretrained(
            str(model_path), local_files_only=True
        )
        model.to(DEVICE)
        model.eval()
        models[name] = (tokenizer, model)
    return models[name]


def predict_proba(texts, tokenizer, model):
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
    with torch.no_grad():
        return torch.softmax(model(**encoded).logits, dim=1).cpu().numpy()


def lime_explanation(text, tokenizer, model, label):
    explainer = LimeTextExplainer(class_names=[LABEL_NAMES[0], LABEL_NAMES[1]], random_state=42)
    explanation = explainer.explain_instance(
        text,
        lambda samples: predict_proba(samples, tokenizer, model),
        labels=[label],
        num_features=10,
        num_samples=LIME_SAMPLES,
    )
    return [
        {"word": word, "weight": round(float(weight), 6)}
        for word, weight in explanation.as_list(label=label)
    ]


def groq_explanation(text, label_name, confidence, lime_items):
    evidence = ", ".join(f"{item['word']} ({item['weight']:+.3f})" for item in lime_items)
    prompt = (
        "Explain a text-classification result clearly and cautiously in 2-3 sentences. "
        "Say this is a model signal, not fact verification. Do not invent facts. "
        f"Verdict: {label_name}; confidence: {confidence:.1f}%; LIME evidence: {evidence}. "
        f"Article excerpt: {text[:3500]}"
    )
    openrouter_key = ""
    if OPENROUTER_KEY_FILE.is_file():
        openrouter_key = OPENROUTER_KEY_FILE.read_text(encoding="utf-8-sig").strip()
    if not openrouter_key or openrouter_key == "PASTE_YOUR_OPENROUTER_API_KEY_HERE":
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        body = json.dumps({
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": "You are a careful explainable-AI assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 180,
        }).encode("utf-8")
        req = url_request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://127.0.0.1:5001",
                "X-Title": "VeriLens Fake News XAI",
            },
            method="POST",
        )
        try:
            with url_request.urlopen(req, timeout=20) as response:
                result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
        except url_error.HTTPError as exc:
            return f"OpenRouter request failed (HTTP {exc.code}); verify the key and OPENROUTER_MODEL."
        except (url_error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
            return f"OpenRouter explanation unavailable ({type(exc).__name__}); review the LIME evidence above."

    api_key = ""
    if GROQ_KEY_FILE.is_file():
        api_key = GROQ_KEY_FILE.read_text(encoding="utf-8-sig").strip()
    if not api_key or api_key == "PASTE_YOUR_GROQ_API_KEY_HERE":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return "Groq explanation unavailable: set GROQ_API_KEY in the environment. The LIME evidence above is still valid."
    body = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a careful explainable-AI assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 180,
    }).encode("utf-8")
    req = url_request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with url_request.urlopen(req, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()
    except url_error.HTTPError as exc:
        if exc.code in {401, 403}:
            return f"Groq rejected the API key (HTTP {exc.code}). Verify that the key is active and copied exactly."
        if exc.code == 400:
            return f"Groq rejected the request (HTTP 400). Verify GROQ_MODEL='{GROQ_MODEL}'."
        return f"Groq request failed (HTTP {exc.code}); review the LIME evidence above."
    except (url_error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
        return f"Groq explanation unavailable ({type(exc).__name__}); review the LIME evidence above."


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    raw_text = payload.get("text", "")
    if not isinstance(raw_text, str):
        return jsonify(error="Article text must be a string."), 400
    text = raw_text.strip()
    name = payload.get("model", "baseline")
    if not isinstance(name, str):
        return jsonify(error="Model must be 'baseline' or 'robust'."), 400
    if not text:
        return jsonify(error="Article text is required."), 400
    try:
        tokenizer, model = load_model(name)
    except (FileNotFoundError, OSError, ValueError) as error:
        return jsonify(error=str(error)), 503

    probabilities = predict_proba([text], tokenizer, model)[0]
    label = int(np.argmax(probabilities))
    confidence = float(probabilities[label] * 100)
    lime_items = lime_explanation(text, tokenizer, model, label)
    return jsonify(
        label=label,
        label_name=LABEL_NAMES[label],
        label_description=LABEL_DESCRIPTIONS[label],
        confidence=confidence,
        probabilities={LABEL_NAMES[index]: round(float(value * 100), 3) for index, value in enumerate(probabilities)},
        lime=lime_items,
        llm_explanation=groq_explanation(text, LABEL_NAMES[label], confidence, lime_items),
        model=f"{name}_model",
        device=str(DEVICE),
        characters=len(text),
    )


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")))
