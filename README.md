# 🇦🇿 Azerbaijani Named Entity Recognition for KYC Tasks

This project focuses on developing a Named Entity Recognition (NER) model specifically for **Know Your Customer (KYC)** tasks in the **Azerbaijani language**. The goal is to automatically extract important entities from customer-related text, such as names, ID numbers, birthdates, email addresses, and more.

---

## 📌 Project Overview

- **Task**: Named Entity Recognition (NER)
- **Language**: Azerbaijani
- **Use Case**: Know Your Customer (KYC)
- **Model**: Fine-tuned BERT (`bert-base-multilingual-cased`)
- **Framework**: Hugging Face Transformers

---

## 🧠 Entities Extracted

The model is trained to identify the following entity types:

- `PERSON`
- `DATE`
- `EMAIL`
- `ID`
- `NATIONALITY`
- `LOCATION`
- `PHONE`
- `ACCOUNT`
- `ORGANIZATION`
- `CARD`
- `TAX_ID`

---

## 📂 Dataset

- The dataset was built manually and consists of tokenized Azerbaijani sentences with corresponding BIO-formatted `ner_tags`.
- Initial data contained annotation errors (e.g. improper BIO tagging), which were cleaned in preprocessing.
- Example:
  ```json
  {
    "tokens": ["menim", "adim", "Maarifdir"],
    "ner_tags": ["O", "O", "B-PERSON"]
  }
  
## 🔧 Preprocessing Steps
# BIO Tag Correction:

Converted all incorrect sequences (e.g., multiple B-PERSON in a row) to proper BIO format.

# Label Alignment:

Used tokenizer's word_ids() to align labels with tokenized subtokens.

Subtokens were assigned -100 to be ignored during loss computation.

## ⚙️ Model Training
 **Pretrained model**: bert-base-multilingual-cased

 **Framework**: Hugging Face Transformers

 **Training method**: Fine-tuning on cleaned dataset

 **Evaluation metrics**: Precision, Recall, F1-score

## 📊 Evaluation Results
Entity	Precision	Recall	F1-score
PERSON	0.01	0.03	0.01
EMAIL	0.41	0.13	0.19
LOCATION	0.05	0.54	0.09
...others	Low scores due to initial label noise and class imbalance		

After correcting entity tags and re-training, performance improved.

## 🧪 How to Run
bash
# Install dependencies
pip install transformers datasets seqeval

# Load and tokenize dataset
python preprocess.py

# Train the model
python train.py

# Evaluate
python evaluate.py
📁 File Structure

├── kyc_ner_dataset_strict_bio.json   # Cleaned BIO dataset
├── kyc_tokenized_aligned_dataset.json
├── preprocess.py
├── train.py
├── evaluate.py
└── README.md

## 📚 Resources
# 📺 YouTube Channels

 - CodeEmporium – BERT for NER
 - Nicholas Renotte
 
# 📄 Articles
- Fine-tuning BERT for NER (HF Blog)

- NER with Transformers (TDS)

- NER with spaCy and BERT

# 🌐 Sites
- Hugging Face Docs

- Kaggle – NER datasets

- Papers With Code – NER

## 👤 Author
# Maarif Sariyev
AI Engineering & NLP Enthusiast
🔗 GitHub: MaarifSariyev

