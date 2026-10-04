# CivicAid AI 🤝

CivicAid AI is a student-built, multilingual public-benefit navigator prototype.

## Problem
People often struggle to discover which government/social-support programs may fit their situation, understand why they match, prepare documents, and avoid unsafe application links.

## Solution
CivicAid combines:
- Profile-based eligibility rules
- TF-IDF/cosine-similarity ML matching
- Explainable match reasons
- Benefit score (prototype)
- Document-readiness checklist
- Official-source safety check
- Human-readable disclaimers

## Run locally

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Important
This is a prototype. Scheme rules and application processes can change. CivicAid is not an official government service and does not make an official eligibility determination. Users should verify current information on the official source before applying.

## Suggested GitHub repo
`civicaid-ai`

## Production roadmap
1. Connect to a verified scheme knowledge base/API.
2. Add multilingual LLM/RAG with citations.
3. Add consent-based document OCR with automatic PII redaction.
4. Add voice input/output for low-literacy users.
5. Add accessibility modes.
6. Add state-wise and district-wise service discovery.
7. Add evaluation dashboard for retrieval accuracy, hallucination rate and task completion.
