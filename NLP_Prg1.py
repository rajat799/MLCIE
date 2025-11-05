# Install dependencies (run once)
# !pip install beautifulsoup4
# !pip install nltk
# !pip install spacy

from bs4 import BeautifulSoup
import nltk, spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
spacy.cli.download("en_core_web_sm")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# --- PART 1: Parse HTML Content ---
html_content = """
<html>
  <head><title>Sample Page</title></head>
  <body>
    <h1>Welcome!</h1>
    <p>This is an example of <b>HTML parsing</b> using BeautifulSoup.</p>
    <a href="https://example.com">Click here</a>
  </body>
</html>
"""

soup = BeautifulSoup(html_content, "html.parser")
text = soup.get_text()
print("Extracted Text:\n", text)

p = soup.find('p').get_text()
print("\nParagraph Only:\n", p)