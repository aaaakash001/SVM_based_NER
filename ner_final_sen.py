from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,fbeta_score
from sklearn.model_selection import KFold
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

# Prepare the features and labels as lists of sentences
X = [sentence['features'] for sentence in dataset_fs['train']]
y = [sentence['ner_tags'] for sentence in dataset_fs['train']]

# Initialize KFold for sentence-level splitting
kf = KFold(n_splits=5, shuffle=True, random_state=seed_value)

# Initialize variables to store sentence-level metrics across folds
# Initialize variables to store fold-level metrics
fold_accuracies, fold_precisions, fold_recalls, fold_f1_scores, fold_f_beta_half_scores, fold_f_beta_two_scores = [], [], [], [], [], []

# KFold Cross-validation loop
for fold, (train_index, test_index) in enumerate(tqdm(kf.split(X), total=kf.n_splits, desc="Training Progress")):
    # Initialize temporary lists for the current fold metrics
    fold_sentence_accuracies, fold_sentence_precisions, fold_sentence_recalls, fold_sentence_f1_scores = [], [], [], []
    fold_sentence_f_beta_half_scores, fold_sentence_f_beta_two_scores = [], [],

    # Split data into training and testing sets based on sentence indices
    X_train_sentences = [X[i] for i in train_index]
    y_train_sentences = [y[i] for i in train_index]
    X_test_sentences = [X[i] for i in test_index]
    y_test_sentences = [y[i] for i in test_index]
    
    # Flatten the data for vectorization
    X_train_flat = [word_features for sentence in X_train_sentences for word_features in sentence]
    y_train_flat = [label for sentence_labels in y_train_sentences for label in sentence_labels]
    X_test_flat = [word_features for sentence in X_test_sentences for word_features in sentence]
    y_test_flat = [label for sentence_labels in y_test_sentences for label in sentence_labels]

    # Vectorize the features for the current fold
    vectorizer = DictVectorizer(sparse=True)
    X_train_vectorized = vectorizer.fit_transform(X_train_flat)
    X_test_vectorized = vectorizer.transform(X_test_flat)

    # Initialize and train the model
    model = svm.SVC(C=2, kernel='linear', gamma='scale', cache_size=2000, verbose=True)
    model.fit(X_train_vectorized, y_train_flat)

    # Predict the labels
    y_pred_flat = model.predict(X_test_vectorized)
    
    # Group predictions and true labels by sentence
    y_true_sentences, y_pred_sentences = [], []
    idx_start = 0
    for sentence_labels in y_test_sentences:
        sentence_length = len(sentence_labels)
        y_true_sentences.append(y_test_flat[idx_start:idx_start + sentence_length])
        y_pred_sentences.append(y_pred_flat[idx_start:idx_start + sentence_length])
        idx_start += sentence_length

    # Calculate metrics at the sentence level
    for true_tags, pred_tags in zip(y_true_sentences, y_pred_sentences):
        accuracy = accuracy_score(true_tags, pred_tags)
        precision = precision_score(true_tags, pred_tags, average='weighted', zero_division=0)
        recall = recall_score(true_tags, pred_tags, average='weighted', zero_division=0)
        f1 = f1_score(true_tags, pred_tags, average='weighted', zero_division=0)
        f_beta_half = fbeta_score(true_tags, pred_tags, average='weighted', zero_division=0, beta=0.5)
        f_beta_two = fbeta_score(true_tags, pred_tags, average='weighted', zero_division=0, beta=2)
        
        # Store metrics for this sentence in fold-specific lists
        fold_sentence_accuracies.append(accuracy)
        fold_sentence_precisions.append(precision)
        fold_sentence_recalls.append(recall)
        fold_sentence_f1_scores.append(f1)
        fold_sentence_f_beta_half_scores.append(f_beta_half)
        fold_sentence_f_beta_two_scores.append(f_beta_two)
    
    # Calculate and store average metrics for this fold
    fold_accuracies.append(np.mean(fold_sentence_accuracies))
    fold_precisions.append(np.mean(fold_sentence_precisions))
    fold_recalls.append(np.mean(fold_sentence_recalls))
    fold_f1_scores.append(np.mean(fold_sentence_f1_scores))
    fold_f_beta_half_scores.append(np.mean(fold_sentence_f_beta_half_scores))
    fold_f_beta_two_scores.append(np.mean(fold_sentence_f_beta_two_scores))
    
    # Print metrics for the current fold
    print(f"Fold {fold + 1} Sentence-Level Results:")
    print(f"Sentence-level Accuracy: {fold_accuracies[-1]:.4f}")
    print(f"Sentence-level Precision: {fold_precisions[-1]:.4f}")
    print(f"Sentence-level Recall: {fold_recalls[-1]:.4f}")
    print(f"Sentence-level F1 Score: {fold_f1_scores[-1]:.4f}")
    print(f"Sentence-level F0.5 Score: {fold_f_beta_half_scores[-1]:.4f}")
    print(f"Sentence-level F2 Score: {fold_f_beta_two_scores[-1]:.4f}\n")
    
    # Confusion Matrix and Classification Report for the current fold
    conf_matrix = confusion_matrix(y_test_flat, y_pred_flat)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", classification_report(y_test_flat, y_pred_flat))

    # Save the model and vectorizer after the first fold (or other criteria as needed)
    if fold + 1 == 5:
        with open('./param/best_ner_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('./param/dict_vectorizer.pkl', 'wb') as vec_file:
            pickle.dump(vectorizer, vec_file)

# Calculate and print average sentence-level metrics across all folds
print("Average Sentence-Level Accuracy: ", np.mean(fold_accuracies))
print("Average Sentence-Level Precision: ", np.mean(fold_precisions))
print("Average Sentence-Level Recall: ", np.mean(fold_recalls))
print("Average Sentence-Level F1 Score: ", np.mean(fold_f1_scores))
print("Average Sentence-Level F0.5 Score: ", np.mean(fold_f_beta_half_scores))
print("Average Sentence-Level F2 Score: ", np.mean(fold_f_beta_two_scores))
