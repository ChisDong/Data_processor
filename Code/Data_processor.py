import pandas as pd
import numpy as np
import unicodedata
import string
import nltk
import re
from pyvi import ViTokenizer
from collections import Counter

nltk.download('punkt')

class Data_Processor:
    def __init__( self, sentimental_path: str, sent_path: str):
        self.sentimental_path = sentimental_path
        self.sent_path = sent_path
        self.stopwords = self.load_stopwords()
        self.abbreviations = self.load_abbreviations()

    def load_stopwords(self):
        try:
            with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as sf: #
                stopwords = [word.strip() for word in sf.readlines()] #delete stopwords in line
            return stopwords
        except FileNotFoundError:
            print(f"Stopwords file 'vietnamese-stopwords.txt' not found.")
            return []
        

    def load_abbreviations(self):
        abbreviations = {}
        try:
            with open('vietnamese_abbreviations.txt', 'r', encoding='utf-8') as af:
                for line in af:
                    abbr, full_form = line.strip().split('=', 1)
                    abbreviations[abbr.strip()] = full_form.strip()
        except FileNotFoundError:
            print(f"Abbreviations file 'vietnamese_abbreviations.txt' not found.")
        return abbreviations
    
    def normalize_abbreviations(self, text):
        for abbr, full_form in self.abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)
        return text
          
    def remove_punctuation(self, text):
        # Use nltk.word  tokenize to tokenize the text
        words = nltk.word_tokenize(text)
        # Remove punctuation
        words = [word for word in words if word not in string.punctuation]
        return ' '.join(words)
        
    def preprocess_text(self, text):
        text = text.lower()
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
        text = ViTokenizer.tokenize(text)  # Tokenization
        text = self.remove_punctuation(text)  # Remove punctuation
        text = ' '.join(word for word in text.split() if word not in self.stopwords)  # Remove stopwords
        return text
    
    def read_txt_files(self): 
        try:
            # read data from file sentiment
            sentiment_data = pd.read_csv(self.sentimental_path, header=None, names=["sentiment"], encoding='utf-8')
            sentiment_data = sentiment_data['sentiment'].tolist()
            
            # read data from file sentences
            sent_data = pd.read_csv(self.sent_path, header=None, names=["sentence"], encoding='utf-8')
            sent_data = sent_data['sentence'].apply(self.preprocess_text).tolist()

            return sentiment_data, sent_data

        except FileNotFoundError:
            print("File not found.")
            return None, None
    
    def create_dataframe(self):
        sentiment_data, sent_data = self.read_txt_files()
        
        if sentiment_data is None or sent_data is None:
            return None
        
        # Count occurrences of each sentiment label
        sentiment_counts = Counter(sentiment_data)

        # Initialize lists for dataframe creation
        text_column = sent_data
        negative_column = [1 if label == 0 else 0 for label in sentiment_data]
        neutral_column = [1 if label == 1 else 0 for label in sentiment_data]
        positive_column = [1 if label == 2 else 0 for label in sentiment_data]

        # Create DataFrame
        df = pd.DataFrame({
            'Text': text_column,
            'Negative': negative_column,
            'Neutral': neutral_column,
            'Positive': positive_column
        })
        return df

