# Changelog

All notable changes to the Devflow NLP Pipeline are documented here.

---

## [1.0.0] — 2026-04-20

### Added
- **Dual-model intent classification**: LinearSVC (TF-IDF) + fine-tuned DistilBERT
- **7 intent classes**: debug, generate, refactor, explain, scaffold, test, document
- **spaCy EntityRuler NER**: extracts LANG, FRAMEWORK, ERROR, COMPONENT entities
- **Structured JSON output** with intent, confidence score, entities, and `low_confidence` flag
- **Synthetic dataset**: 1,260 labelled developer voice utterances (Groq LLaMA 3.3 70B)
- **Evaluation suite**: Macro F1, ablation study, 10-split paired t-test, Cohen's Kappa
- Saved artefacts: `svc_pipeline.pkl`, fine-tuned DistilBERT weights, spaCy pipeline, `label_maps.json`
- Migrated to Google Colab (T4 GPU) for DistilBERT fine-tuning (~10 min)
- Automatic fallback to LinearSVC when DistilBERT is unavailable

### Results
- LinearSVC Macro F1: **0.9948**
- DistilBERT Macro F1: **0.9948** (single split); **0.9990 ± 0.0021** (10 splits)
- Paired t-test p-value: **0.0096** — DistilBERT statistically significantly better
- Cohen's Kappa: **0.8729**

---

## [0.1.0] — 2026-04-14

### Added
- Initial prototype notebook (`Devflow_NLP_Pipeline.ipynb`)
- Dataset generation pipeline using Groq LLaMA 3.3 70B
- Preprocessing pipeline: normalisation, tokenisation, stop-word removal, POS tagging, lemmatisation
- Baseline LinearSVC + TF-IDF classifier
