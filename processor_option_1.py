# contains function related to rep==1 where each topic is extracted and saved in the form
# of a word which best describes the topic.

import json
import operator
from itertools import chain

import pandas as pd
from bson.objectid import ObjectId
from gensim import corpora
from gensim import models
from gensim.models import Phrases
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from pymongo import MongoClient
from transformers import BertTokenizer, BertForSequenceClassification

from get_sentiment import predict_sentiment
from utilities import text_cleaner, get_wordnet_pos


def get_topics(reviews_df, num_topics=5):
    """takes a reviews dataframe and returns the topics extracted by the LDA algorithm"""
    cleaned_text = []
    with open('my_stopwords.txt', 'r') as fp:
        my_stopwords = json.load(fp)

    with open('contraction.json') as f:
        contraction_mapping = json.load(f)

    for t in reviews_df['textContent']:
        cleaned_text.append(text_cleaner(t, 0, my_stopwords, contraction_mapping))

    reviews_df['reviews'] = cleaned_text

    # sentence tokenize
    reviews_df['sentences'] = reviews_df.reviews.apply(sent_tokenize)

    # word tokenize
    reviews_df['tokens_sentences'] = reviews_df['sentences'].apply(
        lambda sentences: [word_tokenize(sentence) for sentence in sentences])

    # pos tagging
    reviews_df['POS_tokens'] = reviews_df['tokens_sentences'].apply(
        lambda tokens_sentences: [pos_tag(tokens) for tokens in tokens_sentences])

    lemmatizer = WordNetLemmatizer()

    # lemmatization
    reviews_df['tokens_sentences_lemmatized'] = reviews_df['POS_tokens'].apply(
        lambda list_tokens_POS: [
            [
                lemmatizer.lemmatize(el[0], get_wordnet_pos(el[1]))
                if get_wordnet_pos(el[1]) != '' else el[0] for el in tokens_POS
            ]
            for tokens_POS in list_tokens_POS
        ]
    )
    reviews_df['tokens'] = reviews_df['tokens_sentences_lemmatized'].map(
        lambda sentences: list(chain.from_iterable(sentences)))

    # converting to bigrams and trigrams
    tokens = reviews_df['tokens'].tolist()
    bigram_model = Phrases(tokens)
    trigram_model = Phrases(bigram_model[tokens], min_count=1)
    tokens = list(trigram_model[bigram_model[tokens]])

    # creating dictionary and model
    dictionary_LDA = corpora.Dictionary(tokens)
    dictionary_LDA.filter_extremes(no_below=2)
    corpus = [dictionary_LDA.doc2bow(tok) for tok in tokens]

    lda_model = models.LdaModel(corpus, num_topics=num_topics,
                                id2word=dictionary_LDA,
                                passes=4, alpha=[0.01] * num_topics,
                                eta=[0.01] * len(dictionary_LDA.keys()))

    topics = lda_model.print_topics(num_words=10)
    extracted_topics = []
    for topic_num in range(len(topics)):
        wp = lda_model.show_topic(topic_num, topn=10)
        for word, prop in wp:
            extracted_topics.append(word)

    return list(set(extracted_topics))


def process_extracted_topics(extracted_topics):
    """takes extracted topics and remove the redundant topics extracted by postprocessing the output"""
    topics = []
    skip = []
    for topic in extracted_topics:
        if "_" in topic:
            temp = topic.split("_")
            if len(temp) == 2 and (temp[0] not in extracted_topics and temp[1] not in extracted_topics):
                topics.append(topic)
            elif len(temp) == 2 and (temp[0] in extracted_topics and temp[1] in extracted_topics):
                topics.append(topic)
                skip.append(temp[0])
                skip.append(temp[1])
            elif len(temp) == 2 and (temp[0] not in extracted_topics and temp[1] in extracted_topics):
                topics.append(topic)
                skip.append(temp[1])
            elif len(temp) == 2 and (temp[0] in extracted_topics and temp[1] not in extracted_topics):
                topics.append(topic)
                skip.append(temp[0])
            else:
                topics.append(topic)

        elif topic not in skip:
            topics.append(topic)
    return topics


def extract_positive_negative(extracted_topics, df, num_topics=5):
    """extracting the positive/negative topics based on the review sentiment"""
    topics = process_extracted_topics(extracted_topics)

    positive = {}
    negative = {}

    for topic in topics:
        positive[topic] = 0

    for topic in topics:
        negative[topic] = 0

    for review, sentiment in zip(list(df.textContent.values), list(df.sentiment.values)):
        # taking the maz score sentiment for the review
        review_sentiment = max(sentiment.items(), key=operator.itemgetter(1))[0]

        for topic in topics:
            if "_" in topic:
                # checking for bigrams and trigrams
                temp = topic.split("_")
                # checking teh counts of positive and negative topics based on sentiment
                if len(temp) == 2 and (temp[0] in review or temp[1] in review) and (
                        review_sentiment.lower()[:3] == 'pos' or review_sentiment.lower()[:3] == 'neu'):
                    positive[topic] += 1
                elif len(temp) == 2 and (temp[0] in review or temp[1] in review) and (
                        review_sentiment.lower()[:3] == 'neg'):
                    negative[topic] += 1
                if len(temp) == 3 and (temp[0] in review or temp[1] in review or temp[2] in review) and (
                        review_sentiment.lower()[:3] == 'pos' or review_sentiment.lower()[:3] == 'neu'):
                    positive[topic] += 1
                elif len(temp) == 3 and (temp[0] in review or temp[1] in review or temp[2] in review) and (
                        review_sentiment.lower()[:3] == 'neg'):
                    negative[topic] += 1
            else:
                if topic in review and (review_sentiment.lower()[:3] == 'pos' or review_sentiment.lower()[:3] == 'neu'):
                    positive[topic] += 1
                elif topic in review and (review_sentiment.lower()[:3] == 'neg'):
                    negative[topic] += 1

    # sorting the topics based on the count attained
    positive = sorted(positive.items(), key=lambda kv: kv[1], reverse=True)
    negative = sorted(negative.items(), key=lambda kv: kv[1], reverse=True)

    positive = [p for p, s in positive[:num_topics]]

    new_neg = []
    for n, s in negative:
        if n not in positive:
            new_neg.append(n)

    negative = new_neg

    negative = [n for n in negative[:num_topics]]
    # returns a list of positive/negative list containing words that best describes the topic in Descending order of the score
    return positive, negative


def get_sentiment_topic(dburl, collection, num_topics=5, output_dir='BERT Fine-Tuning/'):
    """The main function that extracts teh sentiment/topics for all products"""
    client = MongoClient(dburl)
    db = client.get_default_database()
    col = db[collection]

    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    try:
        for ind, product in enumerate(list(col.find())):
            print()
            product_sentiment = {
                "Positive": 0,
                "Negative": 0,
                "Neutral": 0
            }

            print('processing product ' + str(ind))
            print('product_id-', product['_id'])

            if product['reviews']:
                # extracting each review sentiment
                print("\t Extracting product sentiment")
                for i, rev in enumerate(product['reviews']):
                    review_sentiment = predict_sentiment(rev['textContent'], model, tokenizer)

                    product_sentiment["Positive"] += review_sentiment["Positive"]
                    product_sentiment["Negative"] += review_sentiment["Negative"]
                    product_sentiment["Neutral"] += review_sentiment["Neutral"]

                    col.update_one({'_id': product['_id']}, {'$set': {"reviews.%d.sentiment" % i: review_sentiment}})

                no_of_reviews = len(product['reviews'])

                # getting teh average sentiment across review for the product
                for k, v in product_sentiment.items():
                    product_sentiment[k] = float("{:.3f}".format(v / no_of_reviews))

                col.update_one({'_id': product['_id']}, {'$set': {"product_sentiments": product_sentiment}})
                print("\t product_sentiments - %s" % product_sentiment)

                # extracting topics from the reviews
                print("\t Extracting product topics")
                df = pd.DataFrame.from_dict(product['reviews'], orient='columns')
                df = df.dropna(axis=0)
                df['textContent'] = df['textContent'].astype(str)
                topics = {}
                if len(df['textContent'].values) > num_topics:
                    reviews_df = df[['textContent']].copy()
                    extracted_topics = get_topics(reviews_df)
                    positive, negative = extract_positive_negative(extracted_topics, df, num_topics)
                    topics = {
                        "positive": positive,
                        "negative": negative
                    }

                elif 0 < len(df['textContent'].values) < num_topics:
                    reviews_df = df[['textContent']].copy()
                    num_topics = len(df['textContent'].values)
                    extracted_topics = get_topics(reviews_df, num_topics=num_topics)
                    positive, negative = extract_positive_negative(extracted_topics, df, num_topics)
                    topics = {
                        "positive": positive,
                        "negative": negative
                    }
                else:
                    print("\t No review text to process!")
                col.update_one({'_id': product['_id']}, {'$set': {"topics": topics}})
                print("\t topics - %s" % topics)

            else:
                print("\t No reviews to process for this product!")
    except Exception as e:
        print("\t Error %s occured while processing this product!" % str(e))


def get_sentiment_topic_product(dburl, collection, product_id, num_topics=5, output_dir='BERT Fine-Tuning/'):
    """The main function that extracts teh sentiment/topics for a given product id"""
    client = MongoClient(dburl)
    db = client.get_default_database()
    col = db[collection]

    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    try:
        product = col.find_one({'_id': ObjectId(product_id)})
        print()
        product_sentiment = {
            "Positive": 0,
            "Negative": 0,
            "Neutral": 0
        }
        print('processing product')
        print('product_id-', product['_id'])

        if product['reviews']:
            # extracting each review sentiment
            print("\t Extracting product sentiment")
            for i, rev in enumerate(product['reviews']):
                review_sentiment = predict_sentiment(rev['textContent'], model, tokenizer)

                product_sentiment["Positive"] += review_sentiment["Positive"]
                product_sentiment["Negative"] += review_sentiment["Negative"]
                product_sentiment["Neutral"] += review_sentiment["Neutral"]

                col.update_one({'_id': product['_id']}, {'$set': {"reviews.%d.sentiment" % i: review_sentiment}})

            no_of_reviews = len(product['reviews'])

            # getting teh average sentiment across review for the product
            for k, v in product_sentiment.items():
                product_sentiment[k] = float("{:.3f}".format(v / no_of_reviews))

            col.update_one({'_id': product['_id']}, {'$set': {"product_sentiments": product_sentiment}})
            print("\t product_sentiments - %s" % product_sentiment)

            # extracting topics from the reviews
            print("\t Extracting product topics")
            df = pd.DataFrame.from_dict(product['reviews'], orient='columns')
            df = df.dropna(axis=0)
            df['textContent'] = df['textContent'].astype(str)
            topics = {}
            if len(df['textContent'].values) > num_topics:
                reviews_df = df[['textContent']].copy()
                extracted_topics = get_topics(reviews_df)
                positive, negative = extract_positive_negative(extracted_topics, df, num_topics)
                topics = {
                    "positive": positive,
                    "negative": negative
                }

            elif 0 < len(df['textContent'].values) < num_topics:
                reviews_df = df[['textContent']].copy()
                num_topics = len(df['textContent'].values)
                extracted_topics = get_topics(reviews_df, num_topics=num_topics)
                positive, negative = extract_positive_negative(extracted_topics, df, num_topics)
                topics = {
                    "positive": positive,
                    "negative": negative
                }
            else:
                print("\t No review text to process!")
            col.update_one({'_id': product['_id']}, {'$set': {"topics": topics}})
            print("\t topics - %s" % topics)

        else:
            print("\t No reviews to process for this product!")
    except Exception as e:
        print("\t Error %s occured while processing this product!" % str(e))
