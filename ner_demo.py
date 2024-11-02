import numpy as np
import pickle
import streamlit as st
import os
import string
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from sklearn.feature_extraction import DictVectorizer


# Download required NLTK resources (run this once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt_tab')

tag_colors = {
    1: "#8FBC8F",  # Darker Green
    0: "#FF4500", # Darker Tomato
}


pos_tag_map = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
 'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
 'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
 'WP': 44, 'WP$': 45, 'WRB': 46}



# Get the current working directory
current_dir = os.getcwd()

# Define the relative path to the demo_files directory
demo_files_dir = os.path.join(current_dir, 'demo_files')

# Check if we're already in the demo_files directory
if os.path.basename(current_dir) == 'demo_files':
    demo_files_dir = current_dir  # We're already inside demo_files


noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize", "ed", "ing"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ous"]
adv_suffix = ["ward", "wards", "wise", "ly"]
punct = set(string.punctuation)


# Tokenize the sentence
def tokenize_sentence(sentence):
    return word_tokenize(sentence)

# Get parts of speech
def pos_tag_ntlk(tokens):
    return pos_tag(tokens)

# Perform Named Entity Recognition with NLTK
def ner_nltk(sentence):
    tokens = tokenize_sentence(sentence)
    pos_tags = pos_tag_ntlk(tokens)
    named_entities = ne_chunk(pos_tags)
    binary_labels = extract_binary_named_entities(named_entities, tokens)
    return binary_labels

# Function to extract binary named entities
def extract_binary_named_entities(tree, tokens):
    named_entities = { " ".join([word for word, tag in subtree.leaves()]): 1 
                       for subtree in tree if isinstance(subtree, Tree) }
    return [(word, 1 if word in named_entities else 0) for word in tokens]


def word_features(sentence, i):
    word = sentence[i]
    pos_tags = pos_tag_ntlk(sentence)
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

def predict_ner_tags(sentence):
    # Load the saved components
    vectorizer = DictVectorizer(sparse=True)
    with open('./param/dict_vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    with open('./param/best_ner_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    sentence = vectorizer.transform(sentence)
    pred = model.predict(sentence)
    return pred

if __name__ == "__main__":
    st.title("Assignment-3: SVM based Named Entity Recognition (NER)")
    sentence = st.text_input("Enter a sentence:", "Harry Potter was written by J.K. Rowling.")
    if st.button("Predict POS Tags and NER"):
        tokens = tokenize_sentence(sentence)
        
        # NLTK NER results
        nltk_ner_results = ner_nltk(sentence)
        nltk_ner_list = []
        for i, j in nltk_ner_results:
            nltk_ner_list.append(j)
        
        # Custom model predictions
        X_sentence = [word_features(tokens, i) for i in range(len(tokens))]
        predicted_tags = predict_ner_tags(X_sentence)

        # Display the results in Streamlit
        st.write("### NLTK NER Results:")
        
        # Create HTML content for NLTK results
        nltk_html_content = "<div style='display: flex; align-items: flex-start;'>"
        for i in range(len(tokens)):
            word = tokens[i]
            tag = nltk_ner_list[i] if i < len(nltk_ner_list) else 0  # Default to 0 if no tag is found
            word_html = f"<div style='text-align: center; margin: 0 10px;'>"
            word_html += f"<span style='background-color: white; color: black; font-weight: bold; padding: 5px 10px; border-radius: 5px;'>{word}</span><br>"
            word_html += f"<span style='background-color: {tag_colors.get(tag, '#E0E0E0')}; color: white; font-weight: bold; padding: 5px 10px; border-radius: 5px;'>{tag}</span>"
            word_html += "</div>"
            nltk_html_content += word_html
        nltk_html_content += "</div>"
        nltk_html_content += "<br>"
        nltk_html_content += "<br>"
        
        st.markdown(nltk_html_content, unsafe_allow_html=True)

        st.write("### SVM Model NER Result:")
        html_content = "<div style='display: flex; align-items: flex-start;'>"
        for i in range(len(tokens)):
            word = tokens[i]
            tag = predicted_tags[i]
            word_html = f"<div style='text-align: center; margin: 0 10px;'>"
            word_html += f"<span style='background-color: white; color: black; font-weight: bold; padding: 5px 10px; border-radius: 5px;'>{word}</span><br>"
            word_html += f"<span style='background-color: {tag_colors.get(tag, '#E0E0E0')}; color: white; font-weight: bold; padding: 5px 10px; border-radius: 5px;'>{tag}</span>"
            word_html += "</div>"
            html_content += word_html
        html_content += "</div>"
        st.markdown(html_content, unsafe_allow_html=True)
