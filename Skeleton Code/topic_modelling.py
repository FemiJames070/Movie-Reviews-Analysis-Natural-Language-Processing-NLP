# Importing Libraries
import pandas as pd
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load data
data = pd.read_csv("train.csv")

# Function to display the topics
def show_topic(model, feature_names, top):
    """
    Displays the top words for each topic in the LDA model.
    
    Parameters:
        model: Trained LDA model
        feature_names: List of feature names from the vectorizer
        top: Number of top words to display per topic
    """
    for index, distribution in enumerate(model.components_):
        sorted_word_indices = distribution.argsort()[::-1][:top]
        print(f"Topic {index}:")
        print(" ".join([feature_names[i] for i in sorted_word_indices]))
        print()

# Preprocess reviews using spaCy for tokenization, lemmatization, and stopword removal
def preprocess_reviews(reviews):
    """
    Preprocesses the text data by tokenizing, lemmatizing, and removing stopwords and punctuation.
    
    Parameters:
        reviews: List or Series of text data
    
    Returns:
        List of preprocessed text
    """
    processed_reviews = []
    for doc in nlp.pipe(reviews, batch_size=50, disable=["ner", "parser"]):
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        processed_reviews.append(" ".join(tokens))
    return processed_reviews

# Preprocess the reviews
data["cleaned_review"] = preprocess_reviews(data["review"])

# CountVectorizer setup
tf_vectorizer = CountVectorizer(
    max_df=0.95,  # Ignore terms that appear in more than 95% of documents
    min_df=2,     # Ignore terms that appear in fewer than 2 documents
    max_features=1000,  # Limit the vocabulary size
    stop_words="english" # Additional stopword removal
)

# Create the document-term matrix
tf = tf_vectorizer.fit_transform(data["cleaned_review"])

# Latent Dirichlet Allocation setup
lda = LatentDirichletAllocation(
    n_components=5,        # Number of topics
    learning_method="online", # Online learning for large datasets
    random_state=42,       # For reproducibility
    max_iter=10            # Number of iterations
)

# Fit the LDA model
lda.fit(tf)

# Extract feature names and display topics
tf_feature_names = tf_vectorizer.get_feature_names_out()
top = 10
show_topic(lda, tf_feature_names, top)
