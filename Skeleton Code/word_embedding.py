# Import Libraries
from itertools import combinations
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define text for spaCy similarity calculation
text = "funny comedy music laugh humor song songs jokes musical hilarious"
doc = nlp(text)

# Calculate pairwise similarity between tokens in the text using spaCy
print("spaCy Similarity:")
for token1, token2 in combinations(doc, 2):
    print(f"Similarity between '{token1}' and '{token2}': {token1.similarity(token2)}")

# %%
import pandas as pd
from gensim.models import Word2Vec
from tqdm.autonotebook import tqdm

# Load data
data = pd.read_csv("train.csv")

# Preprocess reviews into tokenized sentences for Word2Vec training
def preprocess_for_word2vec(reviews):
    """
    Preprocesses text into tokenized sentences for Word2Vec training.
    
    Parameters:
        reviews: List or Series of text data
    
    Returns:
        List of tokenized sentences
    """
    sentences = []
    for doc in tqdm(nlp.pipe(reviews, batch_size=50, disable=["ner", "parser"])):
        tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
        if tokens:
            sentences.append(tokens)
    return sentences

# Preprocess reviews
tokenized_sentences = preprocess_for_word2vec(data["review"])

# Train a Word2Vec model
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,   # Size of the word vectors
    window=5,          # Context window size
    min_count=2,       # Ignore words that appear less than 2 times
    workers=4,         # Number of worker threads for training
    sg=0               # Use CBOW (set to 1 for skip-gram)
)

# %%
# Calculate similarity between pairs of words in the original text using Word2Vec
print("\nWord2Vec Similarity:")
for token1, token2 in combinations(text.split(), 2):
    # Only calculate similarity if both words are in the vocabulary
    if token1 in model.wv and token2 in model.wv:
        similarity = model.wv.similarity(token1, token2)
        print(f"Similarity between '{token1}' and '{token2}': {similarity:.2f}")
    else:
        print(f"'{token1}' or '{token2}' not in Word2Vec vocabulary.")

