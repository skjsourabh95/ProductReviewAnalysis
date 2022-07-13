# Infant Nutrition-Product reviews analysis
- A cmd tool that reads product reviews from the database and inserts the extracted sentiments and topics from the reviews into each product record

## Tech Stack
- [Anaconda Python3](https://www.anaconda.com/distribution/) or [Python3](https://www.python.org/downloads/)
- [Pytorch](https://pytorch.org/)
- [NLTK](https://www.nltk.org/)
- [Gensim](https://radimrehurek.com/gensim/index.html)
- [MongoDB](https://www.mongodb.com/)
- [Transformers(Bert)](https://github.com/huggingface/transformers)

## Product Dataset
If you have the `products.bson` file already, restore the bson dumb of the db by  -
```cmd
mongorestore --uri="mongodb://localhost:27017/nutrition?authSource=admin" -d nutrition -c products products.bson
```
Otherwise, you can download it from [MongoDB Dump](https://tc-nutrition-data.s3.amazonaws.com/products.zip), extract and run the above command.

## Installing and running locally
Ensure that Pytorch and CUDA (optional for Nvidia GPU acceleration) are installed. 
If CUDA is installed make sure you have the correct pytorch version install which you can find [here](https://pytorch.org/).
Install the correct version by selecting the OS,Package and CUDA. 

Once installed remove these lines from the requirements.txt file -
```
torch==1.4.0+cpu
torchvision==0.5.0+cpu
```
Run the following commands -
```CMD
conda create -n infant python==3.7 -y
conda activate infant
cd /path/to/project/requirements.txt
pip install -r requirements.txt
pip install (pytorch package command) [https://pytorch.org/]
python cli.py
```
This is only required for a certain speed in the execution of the solution. If not available or not working please just follow the below commands.
If CUDA is not available just run the below command it will install all the required packages with pytorch cpu only.

```CMD
conda create -n infant python==3.7 -y
conda activate infant
cd /path/to/project/requirements.txt
pip install -r requirements.txt
python cli.py
```

## cli.py arguments:
- '--help' - Outputs information on all other commands/parameters.
- '--dburl', default=None, MongoDB connection url.
- '--collection', default='products', Name of the products collection.
- '--numtopics', default=5, No of topics to be extracted by the Algorithm
- '--product', default=None, Scans the specified product id, updates the record, and outputs results to stdout
- '--rep', default=1, Represent topics extracted as a (1) word best describing the topic or (2) list of words describing the topic 


## Verification
The command-line tool can be used to verify the entire process of all records being updated.
The reviewer can also get the output of a single product to verify since the solution can take a lot of time to run on all the records.
To run the solution on a single record run the following commands:
```bash
# This will update the record and also output the results for the sentiments/topics to stdout for verification for a single product id.
python cli.py --product=5e7a8abe684a170724f9abcc
```
[Note] - For better understanding of the solution please read solution.md


## Sentiment Analysis
- For the task of sentiment analysis I used BERT with the huggingface PyTorch library to quickly and efficiently fine-tune a model to get near state of the art performance in sentence classification. 
- More broadly, I make use  of transfer learning in NLP to create high performance model with minimal effort on a sentiment analysis task.
- I used BERT to train a text classifier. Specifically, I took the pre-trained BERT model, add an untrained layer of neurons on the end, and train the new model for our classification task.
- I took the inspiration from a well documented google colab notebook which you can find [here](https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX) for training my sentence classifier.
- As the author stated for a low training time , only after 4 epochs(8 hrs) on a Tesla T4, of training I found the model performing really well on my data with good genralization.
- I saved the model into a dir call BERT Fine-Tuning/ from where I call teh trained model and tokenizer for the prediction on new data in the Mongo DB.
- This process could be real slow in a CPU as I found a 160 reviews product took 10 mins to get sentiment for each review and then the average of reviews for the product.
- Thus for the verification I added a method where the review can give teh product id to review and see the results in cmd
```cmd
# This is the current output of the sentiment analyser model-
product_sentiments - {'Positive': 0.828, 'Negative': 0.121, 'Neutral': 0.051}
# This output contains the average confidence scores for the setiments for all the reviews of a product
# Review-wise sentiment for each review could be seen in the mongodb in the reviews field.
```

## Topic Modelling
- For the task of topic modelling since each product doesnt have that many reviews to use a xlnet pre-trained model, I decide to go with a normal LDA with certains steps-
- LDA (short for Latent Dirichlet Allocation) is an unsupervised machine-learning model that takes documents as input and finds topics as output. The model also says in what percentage each document talks about each topic.
- There are 3 main parameters of the model: the number of topics, the number of words per topic, the number of topics per document
- Pre-processing the text, removing punctuations, stopwords, pos-tagging, sentence and word tokenizing with lemmatization.
- To implement the LDA in Python, I use the package gensim.
- Include bi- and tri-grams to grasp more relevant information.
- use only nouns and verbs using POS tagging
- Filtering words that appear in at least 3 (or more) documents is a good way to remove rare words that will not be relevant in topics.
- After a certain bit of fine tuning i was able to make a pipeline that works well on the data and is also very fast to give results.
- This algorithm was inspired by the topic that could be found [here](https://towardsdatascience.com/the-complete-guide-for-topics-extraction-in-python-a6aaa6cedbbc).
- The only doubt i had was regarding how to save the topics positive/negative in the MongoDB. For that i raised a query with co-pilot which can be found [here](https://apps.topcoder.com/forums/?module=Thread&threadID=954157&start=0).
- For this I added a config parameter that is call rep (representation) with option 1 and 2. (Default is option 1 since I found it more helpful)
- Option 1- This extracts the positive/negative topics and saves the word that best represent the topic in the order of the score calculated based on the amount of reviews talking about the topic.
```cmd
# output of the python cli.py --product 5e7a8abe684a170724f9abcc 
 topics - {'positive': ['baby', 'month', 'breast_milk', 'multiple_time', 'love'], 'negative': ['constipation', 'week', 'smell', 'spit', 'fussy']}
 # each word essentially represents a topic.
```
- Option 2- This extracts the positive/negative topics and saves the topics as a list of words that best describes the topic with a  score that is based on no of reviews out of total that have any of the word in the topic.
```cmd
# output of the python cli.py --product 5e7a8abe684a170724f9abcc --rep=2
 topics - {'positive': [(['organic', 'breast_milk', 'highly_recommend', 'price'], 0.5), (['month', 'switch', 'amaze', 'product'], 0.458), (['happy', 'constipation', 'love', 'issue', 'spit'], 0.417), (['feel', 'multiple_time', 'brand', 'bad', 'tell'], 0.333), (['fussy', 'week', 'smell', 'day', 'twin'], 0.208)], 'negative': [(['fussy', 'week', 'smell', 'day', 'twin'], 0.042), (['happy', 'constipation', 'love', 'issue', 'spit'], 0.042)]}
 # each topic essentially represented by a list of words and a score associated with it in terms of sentiments.
```


## cli.py arguments descrition:
- '--dburl', default=None, MongoDB connection url.
```html
The MongoDB url to connect something like -"mongodb://localhost:27017/nutrients"
```
- '--collection', default='products', Name of the products collection.
```html
The MongoDB collection to connect. Default collection name is products and db name is nutrients.
```
- '--numtopics', default=5, No of topics to be extracted by the Algorithm
```html
The no of topics needed to be extracted by the LDA. If the reviews are less adn the topics extracted is less the the required no, 
model takes the no of extracted topics as num_topics.
```
- '--product', default=None, Scans the specified product id, updates the record, and outputs results to stdout
```html
The product id which is the '_id" key in the MongoDB to extract and update the DB for a single product.
Helpful for the reviewer to use this.
```
- '--rep', default=1, Represent topics extracted as a (1) word best describing the topic or (2) list of words describing the topic 
```html
The option 1 or 2 for the rep(representation) decides whether to save teh topics as a word or a list of words as described above.
```

## Verification Process
- To run the solution on a single record run the following commands:
```bash
# This will update the record and also output the results for the sentiments/topics to stdout for verification for a single product id.
python cli.py --product=5e7a8abe684a170724f9abcc
```
- The video of the solution with both options of rep 1/2 for all products could be found [here](https://drive.google.com/file/d/14RxeGiyHeFsPx9878zYk9TAEOuqi0fiI/view?usp=sharing)
- The video of the solution with both options of rep 1/2 for a single products could be found [here](https://drive.google.com/file/d/1UX7k0ZGAqH2Sn9s1SFo7dORwiVLVYRjJ/view?usp=sharing)
