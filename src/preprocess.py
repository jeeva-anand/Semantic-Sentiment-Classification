import re
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):

    text = text.lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()    
    words = [w for w in words if len(w) > 3]
    words = [ps.stem(w) for w in words if w not in stop_words]    
    

    return words

