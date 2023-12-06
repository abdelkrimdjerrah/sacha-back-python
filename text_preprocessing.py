import re
import string
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK resources if necessary
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_data(input_text):
    # Stopword removal
    input_text = " ".join([word for word in str(input_text).split() if word not in stop_words])

    # Lowercasing
    input_text = input_text.lower()

    # URL removal
    input_text = re.sub(r"http\S+|www\S+|https\S+", "", input_text, flags=re.MULTILINE)

    # Punctuation removal
    input_text = input_text.translate(str.maketrans("", "", string.punctuation))

    # Tokenization
    tokens = RegexpTokenizer(r'\w+')
    filtered_list = tokens.tokenize(input_text)

    # Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(w) for w in filtered_list]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]

    return " ".join(lemma_words)
