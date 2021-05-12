import string
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker

nltk.download('stopwords')
checker = SpellChecker()

def remove_punc(s):
    return s.translate(str.maketrans('', '', string.punctuation))

def remove_stop_words(lst):
    words = set(stopwords.words('english'))
    filtered = [word for word in lst if not word in words]
    return filtered

def correct_spelling(words):
    misspelled = checker.unknown(words)

    fixed = []
    for word in words:
        fixed += [word if word not in misspelled else checker.correction(word)]
    return " ".join(fixed)

def process(s):
    s = s.replace("\n", " ")
    filtered = remove_stop_words(s.split(" "))
    corrected = correct_spelling(filtered)
    return remove_punc(corrected.lower())