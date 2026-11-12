
# CS5760 - Natural Language Processing
## Home Assignment 4

### Student Information
- Name: Niharika Mannepalli
- Student ID: 700#775394
- University: University of Central Missouri
- Semester: Fall 2026

## Assignment Overview
This assignment covers sequence-to-sequence models,
attention mechanisms, Transformers, and practical
Natural Language Processing applications.

## Programming Questions

### Q1: NLP Preprocessing Pipeline
File: `nlp_preprocessing.py`

This program tokenizes a sentence, removes common
English stopwords, and applies Porter stemming.

### Q2: Named Entity Recognition
File: `named_entity_recognition.py`

This program uses spaCy to identify named entities
and display their labels and character positions.

### Q3: Scaled Dot-Product Attention
File: `scaled_dot_product_attention.py`

This program uses NumPy to compute attention scores,
apply scaling and softmax, and calculate the final
attention output matrix.

## Requirements
- Python 3
- NLTK
- spaCy
- NumPy
- spaCy English model: en_core_web_sm

## Installation
Run these commands in Terminal:

```bash
python3 -m pip install nltk spacy numpy
python3 -m spacy download en_core_web_sm
```

## Running the Programs

```bash
python3 nlp_preprocessing.py
python3 named_entity_recognition.py
python3 scaled_dot_product_attention.py
```

## Conclusion
This assignment demonstrates fundamental NLP
preprocessing, named entity recognition, and
the scaled dot-product attention mechanism
used in Transformer models.
