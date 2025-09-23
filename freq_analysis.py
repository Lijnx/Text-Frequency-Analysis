import os
import sys
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import pymorphy3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from tqdm.auto import tqdm
from wordcloud import WordCloud


def setup():
    '''
    Настраивает окружение:
    - включает прогресс-бары tqdm для pandas;
    - загружает стоп-слова и токенизатор из NLTK.
    '''
    tqdm.pandas()
    nltk.download('stopwords')
    nltk.download('punkt_tab')


def clean_text(text: str) -> str:
    '''
    Очищает текст:
    - приводит к нижнему регистру;
    - удаляет переносы строк, HTML-теги, URL, пунктуацию, эмодзи;
    - убирает стоп-слова;
    - оставляет только значимые слова.

    Args:
        text (str): исходная строка

    Returns:
        str: очищенный текст
    '''
    text = text.lower()
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r"http\S+", "", text)

    html = re.compile(r'&lt;.*?&gt;')
    text = html.sub(r'', text)

    punctuations = '@#!?+&amp;*[]-%.:/();$=&gt;&lt;|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '')

    sw = stopwords.words('russian')
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    text = " ".join(text)

    emoji_pattern = re.compile(
        "["u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)

    return text


def tokenize(text: str) -> list:
    '''
    Токенизация текста с помощью NLTK.

    Args:
        text (str): строка текста
    
    Returns:
        list: список токенов
    '''
    return word_tokenize(text, language='russian')


def lemmatize(tokens: list) -> list:
    '''
    Лемматизация списка слов с помощью pymorphy3.

    Args:
        tokens (list): список токенов

    Returns:
        list: список лемм
    '''
    morph = pymorphy3.MorphAnalyzer()
    return [morph.parse(token)[0].normal_form for token in tokens]


def save_wordcloud(filename: str, fdist: FreqDist):
    '''
    Генерация и сохранение облака слов.

    Args:
        filename (str): имя выходного файла
        fdist (FreqDist): частотное распределение слов
    '''
    wordcloud = WordCloud(
        width=1920,
        height=1080,
        background_color='white'
    ).generate_from_frequencies(fdist)
    wordcloud.to_file(filename)
    print(f"Облако слов сохранено в {filename}")


def save_bar(filename: str, fdist: FreqDist):
    '''
    Построение и сохранение столбчатой диаграммы
    10 самых частых слов.

    Args:
        filename (str): имя выходного файла
        fdist (FreqDist): частотное распределение слов
    '''
    most_common = fdist.most_common(10)
    words, counts = zip(*most_common)

    plt.figure(figsize=(12, 7))
    plt.bar(words, counts, color="steelblue", edgecolor="black")
    plt.title("10 самых частых слов", fontsize=16)
    plt.xlabel("Слова", fontsize=14)
    plt.ylabel("Частота", fontsize=14)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Диаграмма сохранена в {filename}")


def main():
    '''
    Основная функция:
    - парсит аргументы командной строки;
    - загружает и очищает данные;
    - выполняет токенизацию и лемматизацию;
    - строит и сохраняет wordcloud или диаграмму.
    '''
    setup()

    parser = argparse.ArgumentParser(description="Анализ текстов по частотности слов.")
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Путь к входному файлу (.txt или .tskv)."
    )
    parser.add_argument(
        "-t", "--type", choices=["wordcloud", "bar", "both"], default="wordcloud",
        help="Тип визуализации: 'wordcloud', 'bar' или 'both'."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output.png",
        help="Имя выходного файла (например, result.png)."
    )
    parser.add_argument(
        "-c", "--column", type=str, default="text",
        help="Имя столбца для обработки в .tskv файле (по умолчанию: 'text')."
    )
    args = parser.parse_args()

    # --- проверки ---
    if not os.path.exists(args.input):
        sys.exit(f"Файл не найден: {args.input}")

    if not (args.input.endswith(".tsv") or args.input.endswith(".tskv") or args.input.endswith(".txt")):
        sys.exit("Неподдерживаемый формат. Используйте .txt или .tskv/.tsv")

    # --- загрузка данных ---
    if args.input.endswith((".tsv", ".tskv")):
        with open(args.input, 'r', encoding='utf-8') as f:
            lines = [line.strip().split('\t') for line in f if line.strip() and not line.startswith('#')]
        data = [dict(item.split('=', 1) for item in line if '=' in item) for line in lines]
        df = pd.DataFrame(data)

        if args.column not in df.columns:
            sys.exit(f"Столбец '{args.column}' не найден. Доступные: {list(df.columns)}")

        df = df[[args.column]].rename(columns={args.column: "text"})
    else:  # txt
        with open(args.input, 'r', encoding='utf-8') as f:
            df = pd.DataFrame({"text": f.readlines()})

    if df.empty:
        sys.exit("В файле нет данных для обработки.")

    # --- очистка и обработка текста ---
    print('Очистка текста...')
    df['text'] = df['text'].progress_apply(lambda x: clean_text(x))

    print('Токенизация...')
    df['text'] = df['text'].progress_apply(lambda x: tokenize(x))

    print('Лемматизация...')
    df['text'] = df['text'].progress_apply(lambda x: lemmatize(x))

    # --- частотный словарь ---
    all_words = [word for sublist in df['text'] for word in sublist]
    if not all_words:
        sys.exit("После обработки не осталось слов для анализа.")

    fdist = FreqDist(all_words)

    # --- формирование выходных файлов ---
    base_name = args.output.rsplit('.', 1)[0]

    if args.type == "wordcloud":
        save_wordcloud(args.output, fdist)
    elif args.type == "bar":
        save_bar(args.output, fdist)
    elif args.type == "both":
        save_wordcloud(f"{base_name}_wordcloud.png", fdist)
        save_bar(f"{base_name}_bar.png", fdist)


if __name__ == '__main__':
    main()
