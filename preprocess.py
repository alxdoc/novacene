import pandas as pd
import nltk
import re
import json

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('russian'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

json_file_path = 'dataset.json'

with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

data_list = data["data"]

for item in data_list:
    item['title'] = preprocess_text(item['title'])
    item['description'] = preprocess_text(item['description'])

output_json_file_path = 'processed_dataset.json'
with open(output_json_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
