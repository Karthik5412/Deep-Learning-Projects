import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

tfidf = joblib.load(r'tools\final_tfidf.plk')



# nltk.download('stopwords')
# nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lematizer = WordNetLemmatizer()

def preprocessing_data (text) -> str:

    text = str(text)
    # Converting the text into lower case
    text = text.lower()

    # Removing punctuvations
    text = re.sub(r'[^a-zA-Z0-9\s]','',text)

    # Removing stop words
    words = [w for w in text.split() if w not in stop_words]

    # Lemotization
    words = [lematizer.lemmatize(w) for w in words]

    return ' '.join(words)

def tfidf_vectorize(text) -> list :
    return tfidf.transform(text).toarray()
