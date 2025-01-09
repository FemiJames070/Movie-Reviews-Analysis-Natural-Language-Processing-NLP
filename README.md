# Movie Reviews Analysis and Word Embedding - Natural Language Processing (NLP)
This repository contains multiple lab activities focused on analyzing and processing movie reviews using natural language processing (NLP) and machine learning techniques. 

### The activities include:
1. Activity 1: Processing Movie Reviews
2. Activity 2: Topic Modeling on Movie Reviews
3. Activity 3: Training Word Embedding on Movie Reviews
   
### Requirements
1. Python 3.x
2. spaCy for text processing
3. pandas for data manipulation
4. scikit-learn for machine learning and topic modeling
5. seaborn and matplotlib for visualizations
6. gensim for Word2Vec word embeddings
7. tqdm for progress bars

You can install the required libraries by running:
pip install spacy pandas scikit-learn seaborn matplotlib gensim tqdm

Make sure to download the spaCy model for processing English text:
python -m spacy download en_core_web_sm

## Lab Activities Overview
### Lab Activity 1: Processing Movie Reviews
This activity processes a dataset of movie reviews and performs the following:
1. Extracts adjectives and named entities (PERSON, NORP, FAC, ORG)
2. Counts the number of sentences in each review
3. Visualizes the distribution of sentence counts by sentiment (positive/negative)
4. Displays the most frequent adjectives and entities for each sentiment category

### Lab Activity 2: Topic Modeling on Movie Reviews
In this activity, we perform topic modeling on the movie reviews using Latent Dirichlet Allocation (LDA). It includes the following steps:
1. Preprocessing the reviews (tokenizing, lemmatizing, and stopword removal)
2. Vectorizing the reviews using CountVectorizer
3. Training an LDA model to discover the underlying topics in the reviews
4. Displaying the top words associated with each topic

### Lab Activity 3: Training Word Embedding on Movie Reviews
In this lab, we train word embeddings using the Word2Vec model on movie reviews:
1. Preprocessing the reviews into tokenized sentences
2. Training a Word2Vec model to generate vector representations of words
3. Comparing word similarities using both spaCy and Word2Vec

### File Structure
train.csv: The dataset of movie reviews (for Lab Activities 1 and 2).
Lab_Activity_1.py: Script for processing movie reviews and visualizations.
Lab_Activity_2.py: Script for topic modeling on movie reviews.
Lab_Activity_3.py: Script for training Word2Vec word embeddings.

### How to Run
Clone the repository:
git clone https://github.com/femijames070/movie-reviews-analysis.gitcd movie-reviews-analysis

### Results
The scripts will produce the following outputs:
1. Activity 1: Visualizations showing the distribution of sentence counts and the most frequent adjectives and entities.
2. Activity 2: The top words for each discovered topic using LDA.
3. Activity 3: Similarity scores between pairs of words using spaCy and Word2Vec.

### Prediction
Negative and Postive Review to Test the model

### Evaluation
Accuracy: 84.64%

### Contributions
Feel free to fork the repository, make improvements, and create pull requests.
