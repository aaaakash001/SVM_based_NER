from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, fbeta_score)
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import numpy as np
import pickle
from tqdm import tqdm
from datasets import load_dataset

# Set random seed for reproducibility
seed_value = 42
np.random.seed(seed_value)

# Load the dataset
dataset = load_dataset('conll2003')

# Define suffixes and punctuation
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", 
               "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", 
               "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize", "ed", "ing"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", 
              "ive", "less", "ous"]
adv_suffix = ["ward", "wards", "wise", "ly"]
punct = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

# Function to extract features from words
def word_features(sentence, i, chunk, pos_tags, ner_tags):
    word = sentence[i]
    prevword = sentence[i-1] if i > 0 else '<START>'
    prev_pos_tag = pos_tags[i-1] if i > 0 else -1
    next_pos_tag = pos_tags[i+1] if i < len(sentence)-1 else -1
    nextword = sentence[i+1] if i < len(sentence)-1 else '<END>'
    
    features = {
        'word': word,
        'is_numeric': int(word.isdigit()),
        'contains_number': int(any(char.isdigit() for char in word)),
        'is_punctuation': int(any(char in punct for char in word)),
        'has_noun_suffix': int(any(word.endswith(suffix) for suffix in noun_suffix)),
        'has_verb_suffix': int(any(word.endswith(suffix) for suffix in verb_suffix)),
        'has_adj_suffix': int(any(word.endswith(suffix) for suffix in adj_suffix)),
        'has_adv_suffix': int(any(word.endswith(suffix) for suffix in adv_suffix)),
        'prev_pos_tag': prev_pos_tag,
        'next_pos_tag': next_pos_tag,
        
        'prefix-1': word[:1],
        'prefix-2': word[:2],
        'suffix-1': word[-1:],
        'suffix-2': word[-2:],
        'prevword': prevword,
        'nextword': nextword,
        
        'is_capitalized': int(word[0].isupper()),
        'is_prec_capitalized': int(sentence[i-1][0].isupper() if i > 0 else 0),
        'is_next_capitalized': int(sentence[i+1][0].isupper() if i < len(sentence)-1 else 0),
        
        'is_all_caps': int(word.isupper()),
        'is_all_lower': int(word.islower()),
        
        'word_length': len(word),
        'is_first': int(i == 0),
    }
    return features

# Create features for each example
def create_features(example):
    pos_tags = example['pos_tags'] 
    tokens = example['tokens']
    chunk = example['chunk_tags']
    ner_tags = example['ner_tags']  # Corrected ner_tags extraction
    features = [word_features(tokens, i, chunk, pos_tags, ner_tags) for i in range(len(tokens))]
    return {"features": features, "ner_tags": ner_tags}

# Transform the dataset to include features
dataset_fs = dataset.map(create_features, batched=False)

# Prepare the features and labels as lists of sentences
X = [sentence['features'] for sentence in dataset_fs['train']]
y = [sentence['ner_tags'] for sentence in dataset_fs['train']]

# Take a subset for faster experimentation
X = X[0:5000]
y = y[0:5000]

# Flatten the data for vectorization
X_flat = [word_features for sentence in X for word_features in sentence]
y_flat = [label for sentence_labels in y for label in sentence_labels]

# Vectorize the features
vectorizer = DictVectorizer(sparse=True)
X_vectorized = vectorizer.fit_transform(X_flat)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1,1, 2,3,4,5],
    'kernel': ['linear','rbf'],
    'gamma': ['scale', 'auto'], 
}

# Create the SVC model
svc = svm.SVC()

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=5)
grid_search.fit(X_vectorized, y_flat)

cv_results = grid_search.cv_results_
print(cv_results)

# Use the best model from GridSearch
best_model = grid_search.best_estimator_

# Predict the labels on the whole dataset
y_pred_flat = best_model.predict(X_vectorized)

# Calculate metrics
accuracy = accuracy_score(y_flat, y_pred_flat)
precision = precision_score(y_flat, y_pred_flat, average='weighted', zero_division=0)
recall = recall_score(y_flat, y_pred_flat, average='weighted', zero_division=0)
f1 = f1_score(y_flat, y_pred_flat, average='weighted', zero_division=0)
f_beta_half = fbeta_score(y_flat, y_pred_flat, average='weighted', zero_division=0, beta=0.5)
f_beta_two = fbeta_score(y_flat, y_pred_flat, average='weighted', zero_division=0, beta=2)

# Print the results
print("Best Hyperparameters:", grid_search.best_params_)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"F-beta (beta=0.5): {f_beta_half:.4f}")
print(f"F-beta (beta=2): {f_beta_two:.4f}")

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_flat, y_pred_flat)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_flat, y_pred_flat))

# Save the model and vectorizer
with open('./param/best_ner_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('./param/dict_vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)
