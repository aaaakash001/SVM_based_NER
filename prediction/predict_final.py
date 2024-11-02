
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,fbeta_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import numpy as np
import pickle
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set random seed for reproducibility
seed_value = 42
np.random.seed(seed_value)

# Load the dataset
dataset = load_dataset('conll2003')
print(dataset)

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
    ner_tags = [1 if i > 0 else 0 for i in example['ner_tags']]
    features = [word_features(tokens, i, chunk, pos_tags, ner_tags) for i in range(len(tokens))]
    return {"features": features, "ner_tags": ner_tags}

# Transform the dataset to include features
dataset_fs = dataset.map(create_features, batched=False)
print(dataset_fs)


# Load pre-trained DictVectorizer and model
with open('../param/dict_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)
with open('../param/best_ner_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare the features and labels as lists of sentences
X = [sentence['features'] for sentence in dataset_fs['test']]
y = [sentence['ner_tags'] for sentence in dataset_fs['test']]


# Initialize variables to store sentence-level metrics across folds
sentence_accuracies, sentence_precisions, sentence_recalls, sentence_f1_scores,sentence_f_beta_half_scores,sentence_f_beta_two_scores = [], [], [], [], [], []
incorrect_predictions = []


X_test_flat = [word_features for sentence in X for word_features in sentence]
y_test_flat = [label for sentence_labels in y for label in sentence_labels]


X_test_vectorized = vectorizer.transform(X_test_flat)

# Predict the labels
y_pred_flat = model.predict(X_test_vectorized)

# Group predictions and true labels by sentence
y_true_sentences, y_pred_sentences = [], []
idx_start = 0
for sentence_labels in y:
    sentence_length = len(sentence_labels)
    y_true_sentences.append(y_test_flat[idx_start:idx_start + sentence_length])
    y_pred_sentences.append(y_pred_flat[idx_start:idx_start + sentence_length])
    idx_start += sentence_length
    
    
for true_tags, pred_tags, sentence in zip(y_true_sentences, y_pred_sentences, dataset_fs['test']['tokens']):
        if not np.array_equal(true_tags, pred_tags):
            # Create comma-separated strings of word_tag pairs
            actual_tags_str = ', '.join(f"{word}_{tag}" for word, tag in zip(sentence, true_tags))
            predicted_tags_str = ', '.join(f"{word}_{tag}" for word, tag in zip(sentence, pred_tags))
            # Collect incorrect predictions
            incorrect_predictions.append({
                'sentence': ' '.join(sentence),  # Join words to form the sentence
                'sentence_words_tags_actual': actual_tags_str,
                'sentence_words_tags_predicted': predicted_tags_str,
                'tags_actual': true_tags,
                'tags_predicted': pred_tags
            })
    
# Calculate metrics at the sentence level
for true_tags, pred_tags in zip(y_true_sentences, y_pred_sentences):
    accuracy = accuracy_score(true_tags, pred_tags)
    precision = precision_score(true_tags, pred_tags, average='weighted', zero_division=0)
    recall = recall_score(true_tags, pred_tags, average='weighted', zero_division=0)
    f1 = f1_score(true_tags, pred_tags, average='weighted', zero_division=0)
    f_beta_half = fbeta_score(true_tags, pred_tags, average='weighted', zero_division=0,beta=0.5)
    f_beta_two = fbeta_score(true_tags, pred_tags, average='weighted', zero_division=0,beta=2)

    # Store metrics for this sentence
    sentence_accuracies.append(accuracy)
    sentence_precisions.append(precision)
    sentence_recalls.append(recall)
    sentence_f1_scores.append(f1)
    sentence_f_beta_two_scores.append(f_beta_half)
    sentence_f_beta_half_scores.append(f_beta_two)

# Calculate average metrics for this fold
avg_accuracy = np.mean(sentence_accuracies[-len(y):])
avg_precision = np.mean(sentence_precisions[-len(y):])
avg_recall = np.mean(sentence_recalls[-len(y):])
avg_f1 = np.mean(sentence_f1_scores[-len(y):])
avg_f_beta_half = np.mean(sentence_f_beta_half_scores[-len(y):])
avg_f_beta_two = np.mean(sentence_f_beta_two_scores[-len(y):])

print(f"Sentence-level Accuracy: {avg_accuracy:.4f}")
print(f"Sentence-level Precision: {avg_precision:.4f}")
print(f"Sentence-level Recall: {avg_recall:.4f}")
print(f"Sentence-level F1 Score: {avg_f1:.4f}")
print(f"Sentence-level F1 Score: {avg_f_beta_half:.4f}")
print(f"Sentence-level F1 Score: {avg_f_beta_two:.4f}\n")

# Confusion Matrix and Classification Report for the current fold
conf_matrix = confusion_matrix(y_test_flat, y_pred_flat)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_test_flat, y_pred_flat))


# Save incorrect predictions to a CSV file after all folds
incorrect_df = pd.DataFrame(incorrect_predictions)
incorrect_df.to_csv('incorrect_predictions.csv', index=False)

# Plotting the Confusion Matrix
def plot_confusion_matrix(conf_matrix, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap, 
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

# Plot and save the confusion matrix
class_labels = ["O", "1"]
plot_confusion_matrix(conf_matrix, class_labels)
