import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import pymorphy3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import argparse
import string
import re
from tqdm.auto import tqdm

tqdm.pandas()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
sw = stopwords.words('russian')
morph = pymorphy3.MorphAnalyzer()

def clean_text(text):
    #Приведение текста к нижнему регистру
    text = text.lower()

    # Удаление явных символов переноса строки (особенность датасета)
    text = re.sub(r'\\n', ' ', text)

    # Замена всех не-словесных символов на пробел (кроме букв и знаков препинания)
    text = re.sub(r'\W+', ' ', text)

    # Удаление URL-адресов
    text = re.sub(r"http\S+", "", text)

    # Создание шаблона для HTML-тегов
    html = re.compile(r'&lt;.*?&gt;')

    # Удаление HTML-тегов из текста
    text = html.sub(r'', text)

    # Список пунктуаций для удаления
    punctuations = '@#!?+&amp;*[]-%.:/();$=&gt;&lt;|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '')  # Удаление пунктуации

    # Удаление стоп-слов и приведение слов к нижнему регистру
    text = [word.lower() for word in text.split() if word.lower() not in sw]

    # Объединение слов обратно в текст
    text = " ".join(text)

    # Создание шаблона для поиска эмодзи
    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # эмоции
                        u"\U0001F300-\U0001F5FF"  # символы и пиктограммы
                        u"\U0001F680-\U0001F6FF"  # транспорт и карты
                        u"\U0001F1E0-\U0001F1FF"  # флаги
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)

    # Удаление эмодзи из текста
    text = emoji_pattern.sub(r'', text)

    return text

def tokenize(text):
    # применение метода word_tokenize из библиотеки nltk
    tokenized_text = word_tokenize(text, language='russian')

    return tokenized_text

def lemmatize(text: list) -> list:
    # создание списка лемматизированных слов
    lemmas = [morph.parse(token)[0].normal_form for token in text]

    return lemmas



def main():

    parser = argparse.ArgumentParser(description="???")
    parser.add_argument("-i", "--input", type=str, help="Path to the text file.")
    #parser.add_argument("-o", "--output", type=str, help="Name of an output file.")
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        lines = [line.strip().split('\t') for line in f if line.strip() and not line.startswith('#')]
    data = [dict(item.split('=', 1) for item in line if '=' in item) for line in lines]
    df = pd.DataFrame(data)

    df = df[:10000]
    df = df[['text']]
    print('Cleaning the text...')
    df['text'] = df['text'].progress_apply(lambda x: clean_text(x))

    print('Tokenization...')
    df['text'] = df['text'].progress_apply(lambda x: tokenize(x))

    print('Lemmatization...')
    df['text'] = df['text'].progress_apply(lambda x: lemmatize(x))

    df.info()


if __name__ == '__main__':
    main()