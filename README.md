# Clinical NLP System for Neonatal Discharge Summary Entity Extraction

This project implements a clinical NLP system specifically designed to extract entities from neonatal discharge summaries.

## Entities Extracted

The system identifies and annotates the following entities:
- **P_ID**: Patient identifier
- **Gestational_Age**: Age at birth in weeks
- **Sex**: Gender (Male/Female)
- **Birth_Weight**: Weight at birth
- **Diagnosis**: Medical diagnosis
- **Treatment_Respiratory**: Respiratory treatment information
- **Treatment_Medication**: Medication treatment details
- **Outcome**: Patient outcome/discharge status

## Features

- Rule-based entity extraction with clinical patterns
- Transformer-based NER model option
- Dual output formats: JSON and Markdown table
- Performance evaluation with Precision, Recall, and F1-score
- Comprehensive evaluation framework

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from neonatal_ner import NeonatalNER

# Initialize the model
ner_model = NeonatalNER()

# Extract entities from text
text = "Patient ID: 12345. Male infant, born at 32 weeks gestational age..."
result = ner_model.extract_entities(text)
print(result)
```

## Project Structure

```
model/
├── neonatal_ner.py          # Main NER model
├── entity_patterns.py       # Clinical pattern definitions
├── evaluation.py            # Evaluation framework
├── transformer_model.py     # BERT-based NER model
├── data/                    # Sample data and annotations
├── notebooks/               # Jupyter notebooks for experiments
└── tests/                   # Unit tests
```
