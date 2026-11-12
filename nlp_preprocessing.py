
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download the English stopwords
nltk.download("stopwords")

def preprocess_text(sentence):
    # Step 1: Tokenize the sentence
    tokens = wordpunct_tokenize(sentence)

    # Step 2: Remove English stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        word for word in tokens
        if word.lower() not in stop_words and word.isalpha()
    ]

    # Step 3: Apply stemming
    stemmer = PorterStemmer()
    stemmed_words = [
        stemmer.stem(word) for word in filtered_tokens
    ]

    # Print the results
    print("Original Tokens:", tokens)
    print("Tokens Without Stopwords:", filtered_tokens)
    print("Stemmed Words:", stemmed_words)

# Input sentence from the assignment
sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."

preprocess_text(sentence)
