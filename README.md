# Named Entity Recognition (NER) with SVM

This README provides an overview of the Named Entity Recognition (NER) task using a Support Vector Machine (SVM) model trained with custom word features. The model was built using LibSVM and evaluated with sentence-level performance metrics over 5-fold cross-validation.

---

## Table of Contents
1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Feature Engineering](#feature-engineering)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Installation and Usage](#installation-and-usage)

---

## Project Description

NER is a sub-task of information extraction that involves identifying and classifying named entities (like names, locations, dates, etc.) in text. This project uses a Support Vector Machine (SVM) model to classify words based on features designed to capture structural, lexical, and contextual information.

## Dataset

The dataset used is the `CoNLL-2003` NER dataset, which can be loaded using Hugging Face's `datasets` library. It contains labeled tokens for named entity recognition tasks and includes:
- **Train**: 14,041 samples
- **Validation**: 3,250 samples
- **Test**: 3,453 samples

Each data sample includes:
- `tokens`: words in the sentence
- `pos_tags`: part-of-speech tags for each word
- `chunk_tags`: chunk tags for each word
- `ner_tags`: NER labels for each word

The dataset is transformed to include engineered features for each word.

To load the dataset:
```python
from datasets import load_dataset
dataset = load_dataset('conll2003')
print(dataset)
---

## Feature Engineering

Features are computed for each word based on lexical and contextual characteristics:

| Feature               | Description                                                                                               |
|-----------------------|-----------------------------------------------------------------------------------------------------------|
| `word`                | Original word                                                                                            |
| `is_numeric`          | Binary: 1 if the word is numeric, else 0                                                                 |
| `contains_number`     | Binary: 1 if the word contains any numeric character, else 0                                             |
| `is_punctuation`      | Binary: 1 if the word contains punctuation                                                               |
| `has_noun_suffix`     | Binary: 1 if the word has a noun-like suffix (e.g., '-tion', '-ness')                                    |
| `has_verb_suffix`     | Binary: 1 if the word has a verb-like suffix (e.g., '-ing', '-ed')                                       |
| `has_adj_suffix`      | Binary: 1 if the word has an adjective-like suffix (e.g., '-ous', '-al')                                 |
| `has_adv_suffix`      | Binary: 1 if the word has an adverb-like suffix (e.g., '-ly')                                            |
| `prev_pos_tag`        | POS tag of the previous word                                                                             |
| `next_pos_tag`        | POS tag of the next word                                                                                 |
| `prefix-1`            | First character of the word                                                                              |
| `prefix-2`            | First two characters of the word                                                                         |
| `suffix-1`            | Last character of the word                                                                               |
| `suffix-2`            | Last two characters of the word                                                                          |
| `prevword`            | Previous word in the sentence                                                                            |
| `nextword`            | Next word in the sentence                                                                                |
| `is_capitalized`      | Binary: 1 if the word starts with an uppercase letter                                                    |
| `is_prec_capitalized` | Binary: 1 if the previous word starts with an uppercase letter                                           |
| `is_next_capitalized` | Binary: 1 if the next word starts with an uppercase letter                                               |
| `is_all_caps`         | Binary: 1 if the entire word is uppercase                                                                |
| `is_all_lower`        | Binary: 1 if the entire word is lowercase                                                                |
| `word_length`         | Length of the word                                                                                       |
| `is_first`            | Binary: 1 if the word is the first word in the sentence, else 0                                          |

---

## Model Training

LibSVM is used to train an SVM model on the custom features for each token. The training proceeds in a 5-fold cross-validation setup to evaluate the model’s generalization across multiple subsets of the data. During each fold, the SVM optimization process iterates to find the optimal separating hyperplane.

---

## Evaluation

The model is evaluated at the sentence level with the following metrics:
- **Accuracy**: Proportion of correctly classified sentences
- **Precision**: Proportion of true positive predictions among all positive predictions
- **Recall**: Proportion of true positive predictions among all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **F0.5 Score** and **F2 Score**: Weighted F-scores to account for precision-recall balance

A confusion matrix and classification report are generated to summarize performance on each fold.

---

## Results

The final average metrics across 5 folds are as follows:
- **Sentence-Level Accuracy**: 98.85%
- **Sentence-Level Precision**: 99.27%
- **Sentence-Level Recall**: 98.85%
- **Sentence-Level F1 Score**: 98.95%
- **Sentence-Level F0.5 Score**: 99.11%
- **Sentence-Level F2 Score**: 98.87%

These high scores indicate strong model performance on the NER task, with high accuracy and balanced precision-recall across entities.

---

## Installation and Usage

1. **Install Dependencies**:
   ```bash
   pip install libsvm nltk datasets
