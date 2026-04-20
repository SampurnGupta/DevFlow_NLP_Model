# 🧠 Devflow NLP Pipeline

> **Intent classification & entity extraction for developer voice utterances.**
> A hybrid NLP system that classifies spoken developer commands into structured JSON, combining classical ML (LinearSVC), deep learning (DistilBERT), and rule-based NER (spaCy).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Intent Classes](#intent-classes)
- [Pipeline Architecture](#pipeline-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Artefacts](#artefacts)
- [Dataset](#dataset)
- [Key Design Decisions](#key-design-decisions)
- [Future Work](#future-work)

---

## Overview

Devflow solves the problem of converting raw, noisy developer voice utterances (from Whisper ASR) into structured intent + entity JSON output that can drive an LLM coding assistant.

**Full pipeline:**
```
Voice Input → Whisper ASR → Devflow NLP → Structured JSON → LLM Prompt
```

**Example:**
```
Input:  "um fix the null pointer in the auth service"
Output: {
  "intent":     "debug",
  "confidence": 0.97,
  "entities": {
    "error":     "NullPointerException",
    "component": "auth service"
  },
  "low_confidence": false
}
```

---

## Intent Classes

The pipeline classifies utterances into **7 developer intents**:

| ID | Intent | Example Utterance |
|----|--------|-------------------|
| 0  | `debug` | *"fix the null pointer in login service"* |
| 1  | `generate` | *"write a function that parses JSON"* |
| 2  | `refactor` | *"clean up this authentication module"* |
| 3  | `explain` | *"how does the event loop work"* |
| 4  | `scaffold` | *"create a new Flask project structure"* |
| 5  | `test` | *"add unit tests for the payment service"* |
| 6  | `document` | *"write a docstring for this function"* |

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Preprocessing                          │
│  Text Normalisation → Tokenisation → Stop-word Removal  │
│  → POS Tagging → Lemmatisation                          │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
   ┌──────▼──────┐          ┌───────▼───────┐
   │  LinearSVC  │          │  DistilBERT   │
   │  + TF-IDF   │          │  (fine-tuned) │
   │  (fast)     │          │  (accurate)   │
   └──────┬──────┘          └───────┬───────┘
          └────────────┬────────────┘
                       │
               ┌───────▼────────┐
               │  spaCy NER     │
               │  EntityRuler   │
               │  (rule-based)  │
               └───────┬────────┘
                       │
               ┌───────▼────────┐
               │ Structured JSON │
               │ Output          │
               └────────────────┘
```

**Two inference modes:**
- **Fast mode** — LinearSVC: sub-millisecond inference, deployable anywhere
- **Accurate mode** — DistilBERT: statistically significantly better (p=0.0096), ~10ms/utterance on GPU

If DistilBERT is unavailable, the system automatically falls back to LinearSVC.

**Entity types extracted (spaCy `EntityRuler`):**

| Entity | Examples |
|--------|----------|
| `LANG` | python, javascript, rust |
| `FRAMEWORK` | django, react, flask, pytorch |
| `ERROR` | NullPointerException, TypeError, SegFault |
| `COMPONENT` | auth service, payment module, login handler |

---

## Results

| Metric | Value |
|--------|-------|
| LinearSVC Macro F1 (single split) | **0.9948** |
| DistilBERT Macro F1 (single split) | **0.9948** |
| DistilBERT mean F1 (10 splits) | **0.9990 ± 0.0021** |
| LinearSVC mean F1 (10 splits) | **0.9953 ± 0.0049** |
| Paired t-test p-value | **0.0096** (BERT significantly better) |
| Cohen's Kappa | **0.8729** (substantial agreement) |

> **Key insight:** Both models achieve identical F1 on a single test split due to ceiling performance with a small test set (190 samples). The **paired t-test across 10 randomised splits** reveals DistilBERT is statistically significantly more consistent (lower variance: ±0.0021 vs ±0.0049).

---

## Project Structure

```
NLP/
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── .gitignore
│
├── Devflow_NLP_Pipeline.ipynb          # Initial prototype notebook
├── Devflow_NLP_Pipeline_Final.ipynb    # Final evaluated pipeline (main)
│
├── devflow_dataset.csv                 # Raw training dataset (1,260 utterances)
├── devflow_analysis.md                 # Metric analysis & academic report notes
│
├── Devflow_NLP_Report.docx             # Academic write-up (Word)
├── NLP_ProjectReport.pdf               # Submitted report (PDF)
├── NLP_Project_Report_CSE3.pdf         # CSE3 formatted report (PDF)
│
└── devflow_artefacts/
    ├── devflow_dataset.csv             # Training dataset copy (used in Colab)
    ├── label_maps.json                 # intent↔id mappings (7 classes)
    ├── svc_pipeline.pkl                # Trained LinearSVC + TF-IDF pipeline (~660 KB)
    ├── distilbert/                     # Fine-tuned DistilBERT weights (~256 MB)
    │   ├── model.safetensors
    │   ├── config.json
    │   ├── tokenizer.json
    │   └── tokenizer_config.json
    └── spacy_ner/                      # Trained spaCy EntityRuler pipeline
        ├── config.cfg
        ├── meta.json
        └── [component dirs]
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- 4 GB RAM minimum (8 GB recommended for DistilBERT inference)
- GPU optional but recommended for DistilBERT (Colab T4 works fine)

### Installation

```bash
# Clone / download the project
cd NLP

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (first run only)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### Running the Pipeline

Open `Devflow_NLP_Pipeline_Final.ipynb` in Jupyter or Google Colab and run all cells from top to bottom.

**On Google Colab (recommended for DistilBERT):**
1. Upload `devflow_artefacts/` to Google Drive
2. Mount Drive in Colab
3. Update the `ARTEFACTS_DIR` path in the notebook
4. Run all cells — T4 GPU gives ~10 min for DistilBERT fine-tuning

### Running Inference Only (no training)

If you have the pre-trained artefacts, skip the training sections and jump to the **Section 7: Inference** cell to load and run predictions directly.

---

## Artefacts

> ⚠️ **Model weights are excluded from Git** (see `.gitignore`) due to file size.
> Download or regenerate them using the notebook.

| Artefact | Size | Description |
|----------|------|-------------|
| `svc_pipeline.pkl` | ~660 KB | Serialised `sklearn` Pipeline: TF-IDF + LinearSVC + CalibratedClassifierCV |
| `distilbert/model.safetensors` | ~256 MB | Fine-tuned DistilBERT weights for 7-class intent classification |
| `distilbert/tokenizer.json` | ~695 KB | HuggingFace tokenizer config (tracked in Git) |
| `spacy_ner/` | ~5 MB | spaCy pipeline with custom `EntityRuler` for LANG, FRAMEWORK, ERROR |
| `label_maps.json` | 308 B | intent↔id mappings (tracked in Git) |

---

## Dataset

- **1,260 labelled developer voice utterances** across 7 intent classes (180 per class)
- **Synthetically generated** using Groq LLaMA 3.3 70B with explicit voice-noise injection
- Voice noise features: filler words (`um`, `uh`, `like`), missing punctuation, contractions, ASR-style homophone variants
- Three length buckets: short (~5 words), medium (~12 words), long (~20 words) + noisy variants
- No public dataset exists for developer voice utterances — this is a novel contribution

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Custom stop-word list** | Standard NLTK removes `fix`, `run`, `not` — critical intent signals in developer speech |
| **POS-aware lemmatisation over stemming** | Stemming destroys technical vocab: `authentication → authent`. Lemmatisation preserves it |
| **TF-IDF bigrams** | Unigrams miss compound terms: `null pointer`, `unit test`, `pull request` are essential |
| **`CalibratedClassifierCV` wrapping LinearSVC** | LinearSVC has no native probability output; Platt scaling gives `confidence_score` |
| **DistilBERT over full BERT** | 66M vs 110M params; 40% smaller, 60% faster, 97% BERT performance — avoids overfitting on 1,260 samples |
| **spaCy `EntityRuler` over trained NER** | No gold-standard entity annotations available; rule-based gives 100% precision on known entities |

---

## Future Work

| Priority | Improvement |
|----------|-------------|
| 🔴 High | Test on real Whisper ASR transcriptions of developer speech |
| 🔴 High | Add hard negative examples (utterances between two intents) |
| 🟡 Medium | Increase test set to 300+ samples to reveal true SVC vs BERT gap |
| 🟡 Medium | Add confusion matrix visualisation |
| 🟢 Low | Benchmark `microsoft/codebert-base` vs `distilbert-base-uncased` |
| 🟢 Low | Measure inference latency (ms/utterance) for production profiling |

---

## Academic Context

This project was developed as part of an NLP course (CSE3). It demonstrates **15 NLP techniques** including text normalisation, POS tagging, TF-IDF vectorisation, transfer learning fine-tuning, named entity recognition, ablation studies, and statistical significance testing.

See [`devflow_analysis.md`](devflow_analysis.md) for the full metric analysis and academic report supplement.
