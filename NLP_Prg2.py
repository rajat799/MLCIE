import nltk, spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# Download resources (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
spacy.cli.download("en_core_web_sm")
# Load spaCy model
nlp = spacy.load("en_core_web_sm")
# Setup
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
# Input text
text = input("Enter text:\n")
# Processing
sentences = sent_tokenize(text) # Sentence Tokenization
words = word_tokenize(text) # Word Tokenization
filtered = [w for w in words if w.isalpha() and w.lower() not in stop_words] # Stopword

stemmed = list(map(stemmer.stem, filtered)) # Stemming
lemmatized = [t.lemma_ for t in nlp(" ".join(filtered))] # Lemmatization
# Output
print("\nSentences:", sentences)
print("\nWords:", words)
print("\nFiltered:", filtered)
print("\nStemmed:", stemmed)
print("\nLemmatized:", lemmatized)