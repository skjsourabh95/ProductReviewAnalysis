#!/usr/bin/env python3
import os

import click

from processor_option_1 import get_sentiment_topic, get_sentiment_topic_product
from processor_option_2 import get_sentiment_topic_full, get_sentiment_topic_product_full


@click.command()
@click.option('--dburl', default=None, help="MongoDB connection url.")
@click.option('--collection', default='products', help="Name of the products collection.")
@click.option('--numtopics', default=5, help="No of topics to be extracted.")
@click.option('--product', default=None, help="Read information from provided product id and print to stdout.")
@click.option('--rep', default=1, help="Represent topics extracted as a (1)word or (2)list of words")
def cli(dburl, collection, product, numtopics, rep):
    mongourl = dburl or os.environ.get('MONGO_URL') or "mongodb://localhost:27017/nutrients"
    if rep == 1:
        if product:
            get_sentiment_topic_product(mongourl, collection, product, numtopics)
        else:
            get_sentiment_topic(mongourl, collection, numtopics)
    elif rep == 2:
        if product:
            get_sentiment_topic_product_full(mongourl, collection, product, numtopics)
        else:
            get_sentiment_topic_full(mongourl, collection, numtopics)


if __name__ == '__main__':
    cli()
