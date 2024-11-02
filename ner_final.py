from sklearn.metrics import (accuracy_score, f1_score, fbeta_score, 
                             recall_score, precision_score, 
                             confusion_matrix, classification_report)
from sklearn import svm
import numpy as np
import pandas as pd
import string
import pickle
import random
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm
from datasets import load_dataset

# Set random seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

# Load the dataset
dataset = load_dataset('conll2003')
print(dataset)

# Define suffixes
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", 
               "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", 
               "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize", "ed", "ing"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", 
              "ive", "less", "ous"]
adv_suffix = ["ward", "wards", "wise", "ly"]
punct = set(string.punctuation)

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
    ner_tags = [1 if i > 0 else 0 for i in example['ner_tags']]
    feature = [word_features(tokens, i, chunk, pos_tags, ner_tags) for i in range(len(tokens))]
    return {"features": feature, "ner_tags": ner_tags}

# Transform the dataset
dataset_fs = dataset.map(create_features, batched=False)
print(dataset_fs)

# Prepare the features and labels
X = []
y = []
for sentence in dataset_fs['train']:
    X.append(sentence['features'])
    y.append(sentence['ner_tags'])

# Flatten the features and labels
X_flat = [word_features for sentence in X for word_features in sentence]
y_flat = [label for sentence_labels in y for label in sentence_labels]

# X_flat = X_flat[:50000]
# y_flat = y_flat[:50000]
assert len(y_flat) == len(X_flat)

# Vectorize the features
vectorizer = DictVectorizer(sparse=True)
X_vectorized = vectorizer.fit_transform(X_flat)

# Define number of folds for KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed_value)

# Initialize variables for storing metrics
accuracies, precisions, recalls, f1_scores = [], [], [], []
fbeta_half_scores, fbeta_two_scores = [], []
all_y_true, all_y_pred = [], []


# KFold Cross-validation
for fold, (train_index, test_index) in enumerate(tqdm(kf.split(X_vectorized), total=kf.n_splits, desc="Training Progress")):
    # Split data into training and testing sets
    X_train, X_test = X_vectorized[train_index], X_vectorized[test_index]
    y_train, y_test = np.array(y_flat)[train_index], np.array(y_flat)[test_index]
    
    # Initialize and train the model
    model = svm.SVC(C=2.0, kernel='Linear', gamma='scale', cache_size=2000)  # Setting cache size to 500 MB

    model.fit(X_train, y_train)

    # Predict the labels
    y_pred = model.predict(X_test)

    # Append true and predicted labels
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    # Calculate metrics for this fold
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    fbeta_half = fbeta_score(y_test, y_pred, beta=0.5, average='weighted', zero_division=0)
    fbeta_two = fbeta_score(y_test, y_pred, beta=2, average='weighted', zero_division=0)

    # Store metrics
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    fbeta_half_scores.append(fbeta_half)
    fbeta_two_scores.append(fbeta_two)

    # Print metrics for the current fold
    print(f"Fold {fold + 1} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F-beta Score (beta=0.5): {fbeta_half:.4f}")
    print(f"F-beta Score (beta=2): {fbeta_two:.4f}\n")
    
    # Save the model after the first fold (or implement your criteria)
    if fold + 1 == 1:
        with open('./param/best_ner_model_pos_feature.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        # Save the DictVectorizer after fitting on the training data
        with open('./param/dict_vectorizer.pkl', 'wb') as vec_file:
            pickle.dump(vectorizer, vec_file)


# Calculate average metrics over all folds
print("Average Accuracy: ", np.mean(accuracies))
print("Average Precision: ", np.mean(precisions))
print("Average Recall: ", np.mean(recalls))
print("Average F1 Score: ", np.mean(f1_scores))
print("Average F-beta Score (beta=0.5): ", np.mean(fbeta_half_scores))
print("Average F-beta Score (beta=2): ", np.mean(fbeta_two_scores))

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(all_y_true, all_y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(all_y_true, all_y_pred))
