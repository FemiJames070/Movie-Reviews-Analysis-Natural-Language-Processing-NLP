# Importing Libraries
import pandas as pd
import spacy
from tqdm import tqdm
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load data
data = pd.read_csv("train.csv")

# Function to extract adjectives from a review
def get_adjs(doc):
    return [token.text for token in doc if token.pos_ == "ADJ"]

# Function to extract entities (PERSON, NORP, FAC, ORG) from a review
def get_ents(doc):
    return [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "NORP", "FAC", "ORG"}]

# Function to count the number of sentences in a review
def get_number_of_sents(doc):
    return len(list(doc.sents))

# Initialize counters and lists
number_of_sents, pos_adj_counter, neg_adj_counter, ent_counter = (
    [],
    Counter(),
    Counter(),
    Counter(),
)

# Analyze reviews
for review, sentiment in tqdm(data.to_records(index=False), desc="Processing reviews"):
    doc = nlp(review)

    # Adjectives, Entities, and Sentence Count
    adjs = get_adjs(doc)
    ents = get_ents(doc)
    num_sents = get_number_of_sents(doc)
    number_of_sents.append(num_sents)

    # Update counters
    ent_counter.update(ents)
    if sentiment == "positive":
        pos_adj_counter.update(adjs)
    elif sentiment == "negative":
        neg_adj_counter.update(adjs)

# Add sentence counts to the DataFrame
data["number_of_sents"] = number_of_sents

# %%
# Boxplot: Number of sentences grouped by sentiment
sns.set(style="whitegrid")
sns.boxplot(x="sentiment", y="number_of_sents", data=data)
plt.title("Number of Sentences by Sentiment")
plt.show()

# %%
# Bar plot: Top 20 most frequent entities
top20_ent = ent_counter.most_common(20)
y = [t[0] for t in top20_ent]
x = [t[1] for t in top20_ent]

sns.barplot(x=x, y=y)
plt.title("Top 20 Most Frequent Entities")
plt.xlabel("Frequency")
plt.ylabel("Entities")
plt.show()

# %%
# Bar plots: Top 20 adjectives in positive and negative reviews
_, axes = plt.subplots(1, 2, figsize=(16, 8))

top20_pos_adj = pos_adj_counter.most_common(20)
y_pos = [t[0] for t in top20_pos_adj]
x_pos = [t[1] for t in top20_pos_adj]
sns.barplot(x=x_pos, y=y_pos, ax=axes[0], palette="Greens_r")
axes[0].set_title("Top 20 Adjectives in Positive Reviews")
axes[0].set_xlabel("Frequency")
axes[0].set_ylabel("Adjectives")

top20_neg_adj = neg_adj_counter.most_common(20)
y_neg = [t[0] for t in top20_neg_adj]
x_neg = [t[1] for t in top20_neg_adj]
sns.barplot(x=x_neg, y=y_neg, ax=axes[1], palette="Reds_r")
axes[1].set_title("Top 20 Adjectives in Negative Reviews")
axes[1].set_xlabel("Frequency")
axes[1].set_ylabel("Adjectives")

plt.tight_layout()
plt.show()


