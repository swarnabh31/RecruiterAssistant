# AI Recruiter Assistant

A local-first, privacy-focused Streamlit application that helps recruiters analyze job descriptions, evaluate resumes, and perform AI-powered market research — powered entirely by local LLMs via [Ollama](https://ollama.ai). No cloud APIs, no data leaves your machine (except live web searches performed by the research agent).

---

## ✨ Features

### 📋 JD Analysis
Upload a job description (PDF, DOCX, or TXT) and get a structured, easy-to-read breakdown:
- Role summary
- Must-have vs. good-to-have skills
- Experience required
- Target company types to source from
- **LinkedIn X-ray search strings** — ready-to-click Google searches scoped to `site:linkedin.com/in/`, so you can find matching candidate profiles in one click
- 10 tailored interview questions
- A ready-to-send outreach email template

Results render as clean sections with headings, bullet points, and progress indicators — not raw text. Export to Markdown or JSON any time.

The tab also includes a **Chat with JD** sub-tab — ask follow-up questions about the active job description, clarify requirements, brainstorm sourcing strategy, or get advice on screening — in a normal chat interface with streaming responses. Download the full chat log as Markdown.

### 📄 Resume Evaluation (Batch)
Upload **multiple resumes at once** and evaluate them all against the active JD in one pass. Each candidate gets:
- Overall suitability score (0–100) with a visual progress bar
- Skills match breakdown (must-have present/missing, good-to-have present)
- Experience analysis (relevant years, company-type match, project complexity)
- Red flags (employment gaps, job hopping, etc.)
- Key strengths and weaknesses
- A clear Yes / No / Maybe hiring recommendation with reasoning

Results are automatically **ranked by score**, shown as expandable cards, and exportable as a single CSV (great for sharing with a hiring manager) or combined Markdown report.

### 🕵️ Agentic Research Lab
An AI-powered research assistant with live web search, document memory, and streaming chat — **works independently, no JD required.**

- **Knowledge Base** — upload documents (PDF, DOCX, TXT) via the sidebar to inject them into the agent's memory. The agent can reference these alongside live search results.
- **Intelligent Chat** — ask questions, request structured market reports ("Executive Summary, Financials, Trends, Conclusion"), or query your uploaded documents. The agent autonomously uses the `perform_web_search` tool when it needs current information.
- **Live Tool Visibility** — the agent shows its work in real time ("🔍 Searching Web: `...`"), so you always know what it's looking up.
- **Report Formatting** — long responses and market reports are rendered in a clear monospace style for easy reading.

### 🗂️ Session History
Every JD you analyze is automatically saved locally (SQLite — no cloud database). Reload any past session from the sidebar to pick up exactly where you left off, JD and all.

---

## 🖥️ Prerequisites

1. **[Ollama](https://ollama.ai)** installed and running on your machine (or accessible over your network).
2. At least one Ollama model pulled that's good at following instructions, e.g.:
   ```bash
   ollama pull llama3.2
   ```
3. **For the Agentic Research Lab** (which uses tool-based web search), you need a model that supports **tool calling**. Not all models do. Check with:
   ```bash
   ollama show <model_name>
   ```
   and look for `tools` under `Capabilities`. `qwen2.5`, `qwen3.6`, and most `llama3.x` models support this well.
4. Python 3.10+ and an internet connection (only needed for the Agentic Research Lab feature — everything else is fully offline).

---

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/swarnabh31/RecruiterAssistant.git
   cd RecruiterAssistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is running and has at least one model available:
   ```bash
   ollama serve
   ollama pull llama3.2
   ```

4. Launch the app:
   ```bash
   streamlit run app.py
   ```

5. Open the URL Streamlit gives you (usually `http://localhost:8501`) in your browser.

---

## ⚙️ Configuration (optional)

The app works out of the box with sensible defaults, but you can customize it via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Point the app at a different Ollama host (e.g. if Ollama runs on another machine on your network) |

Set it before launching, e.g. on Windows PowerShell:
```powershell
$env:OLLAMA_BASE_URL = "http://192.168.1.100:11434"
streamlit run app.py
```

---

## 📖 How to Use It

### Analyzing a job description
1. Upload your JD in the **sidebar** (this makes it "active" across every tab — no need to re-upload it per feature).
2. Go to the **JD Analysis** tab and click **Analyze Job Description**.
3. Review the structured breakdown. Click any LinkedIn X-ray search's **"Search →"** button to open it directly in Google, scoped to LinkedIn profiles.
4. Download the analysis as Markdown or JSON if you want to keep it.

### Evaluating candidates
1. With a JD active, go to the **Resume Evaluation** tab.
2. Upload one or many resumes at once.
3. Click **Evaluate All Resumes** — a progress bar shows you how far along it is.
4. Results appear ranked by score, best candidate first. Expand any card for the full breakdown.
5. Export the shortlist as CSV to share with your team, or Markdown for a written report.

### Using the Agentic Research Lab
1. Go to the **Agentic Research Lab** tab — no JD required.
2. (Optional) Upload documents to the **Knowledge Base** in the sidebar to give the agent reference material.
3. Ask questions, request market reports, or query your documents via the chat interface.
4. The agent will search the web automatically when it needs current information — you'll see each search live.

### Picking up where you left off
Use the **History** panel in the sidebar to reload any previously analyzed JD and its saved session.

---

## 🔒 Privacy & Local-First Design

- All document analysis, resume evaluation, and chat runs entirely on **your own machine** through Ollama — nothing is sent to a third-party AI provider.
- Session history and evaluation results are stored in a local SQLite file (`recruiter_assistant.db`), never uploaded anywhere.
- The **only** features that reach the internet are the Agentic Research Lab and JD Analysis LinkedIn X-ray links (which open Google in your browser) — live web searches (via DuckDuckGo) are performed to stay current. No resume, candidate, or JD content is ever sent as part of those searches.

---

## 🛠️ Troubleshooting

**"Ollama is not running or no models found"**
Make sure `ollama serve` is running and you've pulled at least one model with `ollama pull <model>`.

**Agentic Research Lab errors or the agent doesn't call any tools**
Your selected model likely doesn't support tool calling. Run `ollama show <model>` and confirm `tools` appears under Capabilities — if not, switch to a model that supports it.

**A PDF/DOCX shows "This looks like a scanned document"**
The file likely has no extractable text layer (e.g. it's a scanned image). Consider running it through OCR software first, or re-export the original as a text-based PDF.

**Analysis takes a long time**
This is expected with larger local models — generation speed depends on your GPU/CPU. Smaller/quantized models will respond faster at some cost to output quality.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
