from gensim.models import *
from utils.dataset import *
import nltk
from string import digits



def main():
    d = DataSet(name="train")
    stances = d.stances
    articles = d.articles
    train_headlines_model1, train_article_bodies_model1, train_labels_model1 = tokenize_dataset(stances, articles, "model1")
    related_stances = split_related_and_unrelated(stances)
    train_w2v_model(train_headlines_model1, train_article_bodies_model1)
    # train_headlines_model2, train_article_bodies_model2, train_labels_model2 = tokenize_dataset(related_stances, articles, "model2")


def tokenize_dataset(stances, articles, model):
    train_headlines = []
    train_article_bodies = []
    train_labels = []

    print("tokenizing headlines...")
    idx = 0
    for s in stances:
        idx+=1
        print("process: " + str(idx) + "/" + str(len(stances)))
        headline = s['Headline']
        id = s['Body ID']
        stance = s['Stance']
        if model == "model2":
            if stance == "agree":
                train_labels.append(0)
            elif stance == "disagree":
                train_labels.append(1)
            else:
                train_labels.append(2)

        elif model == "model1":
            if stance == "unrelated":
                train_labels.append(0)
            else:
                train_labels.append(1)

        # tokenizing headlines
        headline = headline.lower()
        table = str.maketrans('', '', digits)
        headline = headline.translate(table)
        words = nltk.word_tokenize(headline)
        headline_tokenized = [word for word in words if word.isalnum()]
        headline_tokenized_no_sw = [word for word in headline_tokenized if not word in nltk.corpus.stopwords.words()]
        train_headlines.append(headline_tokenized_no_sw)

    # tokenizing articles
    idx = 0
    print("tokenizing articles...")
    for article in list(articles.values()):
        idx+=1
        print("process: " + str(idx) + "/" + str(len(stances)))
        article = article.lower()
        table = str.maketrans('', '', digits)
        article = article.translate(table)
        words = nltk.word_tokenize(article)
        article_tokenized = [word for word in words if word.isalnum()]
        article_tokenized_no_sw = [word for word in article_tokenized if not word in nltk.corpus.stopwords.words()]
        train_article_bodies.append(article_tokenized_no_sw)

    print("train headlines size " + str(len(train_headlines)))
    print("train article bodies size " + str(len(train_article_bodies)))
    max_headline = max(len(h) for h in train_headlines)
    max_article = max(len(a) for a in train_article_bodies)
    print("maximum headlines " + str(max_headline))
    print("maximum articles " + str(max_article))

    return train_headlines, train_article_bodies, train_labels


def split_related_and_unrelated(stances):
    related_stances = []
    for s in stances:
        if s['Stance'] != "unrelated":
            related_stances.append(s)

    return related_stances

def word_embeddings(stances):
    directory = "./GoogleNews-vectors-negative300.bin"
    model = KeyedVectors.load_word2vec_format(directory, binary=True)
    word_vectors = []
    for s in stances:
        vectors = []
        for w in s:
            vector = model[w]
            vectors.append(vector)
        word_vectors.append(vectors)

    print(len(word_vectors))
    print(word_vectors)


def train_w2v_model(stances, articles):
    addtional_signs = ["<sos>", "<unk>", "<pad>", "<eos>"]
    corpus = stances + articles + addtional_signs
    print("Start training")
    model = Word2Vec(corpus, min_count=1, workers=8)
    model.save('./w2v.model')

if __name__ == "__main__":
    main()
