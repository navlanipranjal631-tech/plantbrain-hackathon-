# 🏭 PlantBrain — Unified Industrial Knowledge Intelligence

**ET AI Hackathon 2026 — Problem Statement #8**
*AI for Industrial Knowledge Intelligence: Unified Asset & Operations Brain*

Built by **Pranjal Navlani** — B.Sc. Bioinformatics, Statistics and Computer Science, REVA University

---

## The Problem

Large industrial plants operate across **7–12 disconnected document systems** — maintenance logs, safety procedures, inspection reports, and regulations — each maintained separately. Engineers spend up to **35% of their working hours** just searching for information that already exists somewhere in the organisation (McKinsey, 2024). India recorded over **6,500 fatal workplace accidents in FY2023** (DGFASLI) — a pattern rooted not in a lack of data, but in a lack of connected intelligence. Warning signals exist; they simply don't reach the right decision in time.

## The Solution

**PlantBrain** unifies scattered plant documents into a single AI-queryable knowledge layer with three integrated capabilities:

| Feature | What it does |
|---|---|
| 💬 **Ask PlantBrain** | Retrieval-Augmented Generation (RAG) — ask any question about maintenance, safety, or compliance and get an answer in seconds, with the exact source document cited every time. |
| 🕸️ **Knowledge Graph** | Automatically extracts equipment tags, work orders, and regulation references from every document (regex-based, zero API cost) and links them into an interactive network — surfacing cross-document connections a manual search would miss. |
| 🛡️ **Compliance Checker** | Cross-references real-world operational scenarios against procedures, regulations, and historical near-miss records to automatically flag violations — before they become an audit finding or an accident. |

## Architecture

```
Plant Documents → Embedding Model (Gemini) → Vector Store (disk-cached)
                                                     │
                              User Question ─────────┤
                                                     ▼
                                          Similarity Search
                                          (cosine similarity)
                                            │            │
                                            ▼            ▼
                                    Answer Engine   Compliance Checker
                                     (Groq/Llama)      (verdict + reasoning)
                                            │            │
                                            └─────┬──────┘
                                                   ▼
                                          PlantBrain Console

  Knowledge Graph runs in parallel — pattern-matching entity extraction,
  no API calls required.
```

## Tech Stack

- **Frontend:** Streamlit (Python)
- **Embeddings:** Google Gemini (`gemini-embedding-001`)
- **Answer Generation:** Groq (Llama 3.3 70B), with Gemini fallback
- **Vector Search:** NumPy cosine similarity, disk-cached embeddings
- **Knowledge Graph:** NetworkX + PyVis (regex-based entity extraction)
- **Retrieval Method:** Retrieval-Augmented Generation (RAG)

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/plantbrain-hackathon.git
cd plantbrain-hackathon

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Add your API keys to a .env file
echo "GOOGLE_API_KEY=your_key_here" >> .env
echo "GROQ_API_KEY=your_key_here" >> .env

# 5. Run the app
python -m streamlit run app.py
```

Free API keys: [Google AI Studio](https://aistudio.google.com) · [Groq Console](https://console.groq.com)

## Project Structure

```
plantbrain-hackathon/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── data/                   # Synthetic plant document corpus (20 files)
├── .gitignore
└── README.md
```

## Why This Matters

It is easy to discuss industrial safety in the language of compliance scores and audit checklists. But the people most exposed to these risks are rarely the executives reading those reports — they are shift workers, contract laborers, and maintenance technicians, often without the standing to question a rushed permit or a skipped inspection step. PlantBrain is built on the premise that the information needed to prevent a fatality should reach the person who needs it in seconds — not just the person who can afford a dashboard.

## Judging Criteria Alignment

| Criteria | How PlantBrain Addresses It |
|---|---|
| **Innovation** | Two distinct agents sharing one knowledge base, plus a zero-cost, deterministic knowledge graph. |
| **Business Impact** | Directly reduces the 35% of engineer time lost to manual document search. |
| **Technical Excellence** | Working RAG pipeline with source citation, disk-cached embeddings, structured compliance verdicts. |
| **Scalability** | Ingestion, embedding cache, and entity extraction scale linearly with corpus size. |
| **User Experience** | Guided, plain-language interface explaining *why* each feature exists, not just *what* it does. |

---

*This prototype is built entirely on synthetic data modeled on realistic plant operations, designed to generalise directly to a real facility's document corpus.*
