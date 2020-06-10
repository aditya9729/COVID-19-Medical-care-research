
from collections import Counter

from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim import corpora, models
import gensim

import pyLDAvis
import pyLDAvis.gensim
import warnings
import pickle

from pprint import pprint

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import os
from nltk.stem.wordnet import WordNetLemmatizer
stop = stopwords.words('english')

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def text_cleaner(Series):
    
    Series=Series.dropna()
    
    Series=Series.apply(lambda x: " ".join(word.lower() for word in str(x).split()))
    Series=Series.apply(lambda x: " ".join(word for word in str(x).split() if word not in stop))
    Series=Series.str.replace('[^\w\s]','')
    Series=Series.apply(lambda x: " ".join(word for word in str(x).split() if len(word)>3))
    Series=Series.apply(lambda x: " ".join(get_lemma(word) for word in str(x).split()))
    Series=Series.apply(lambda x:" ".join(word for word in str(x).split() if word.isalpha()))
    
    months=['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    
    Series=Series.apply(lambda x:" ".join(word for word in str(x).split() if word not in months))
    
    
    unigrams = Series.apply(word_tokenize)
    bigram_phrases = Phrases(unigrams)
        
    bigram_phrases = Phraser(bigram_phrases)
    
    sentences_bigrams_filepath = os.path.join(os.getcwd(), str(Series.name)+'_sentence_bigram_phrases_all.txt')
    
    with open(sentences_bigrams_filepath, 'w') as f:
        
        for sentence_unigrams in tqdm.tqdm(unigrams):
            
            sentence_bigrams = ' '.join(bigram_phrases[sentence_unigrams])
            
            f.write(sentence_bigrams + '\n')
    sentences_bigrams = LineSentence(sentences_bigrams_filepath)
    
    for sentence_bigrams in tqdm.tqdm(it.islice(sentences_bigrams, 60, 70)):
        print(' '.join(sentence_bigrams))
        print('')
    
    trigram_phrases = Phrases(sentences_bigrams)
    
    # Turn the finished Phrases model into a "Phraser" object,
    # which is optimized for speed and memory use
    trigram_phrases = Phraser(trigram_phrases)
    
    sentences_trigrams_filepath = os.path.join(os.getcwd(),str(Series.name)+ '_sentence_trigram_phrases_all.txt')
    with open(sentences_trigrams_filepath, 'w') as f:
        
        for sentence_bigrams in tqdm.tqdm(sentences_bigrams):
            
            sentence_trigrams = ' '.join(trigram_phrases[sentence_bigrams])
            
            f.write(sentence_trigrams + '\n')
            
    sentences_trigrams = LineSentence(sentences_trigrams_filepath)
    
    for sentence_trigrams in tqdm.tqdm(it.islice(sentences_trigrams, 60, 70)):
        print(' '.join(sentence_trigrams))
        print('')
        
    return sentences_trigrams_filepath


## Use any series// Change series here
abstract_path=text_cleaner(meta_df['abstract'])

def bow_generator(filepath):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """
    
    for review in LineSentence(filepath):
        yield dictionary_trigrams.doc2bow(review)

def topic_modeling(Series_path):
    series_list=LineSentence(Series_path)

    lists=[]
    for item in series_list:
        lists.append(item)
        
    tokens=word_tokenize(' '.join(word for item in lists for word in item))
    
    print(Counter(tokens).most_common(50))
    
    reviews_trigrams = LineSentence(abstract_path)

    # learn the dictionary by iterating over all of the reviews
    dictionary_trigrams = Dictionary(reviews_trigrams)
    
    bow_corpus_filepath = os.path.join(os.getcwd(), 'bow_trigrams_corpus_all.mm')
    
    MmCorpus.serialize(
        bow_corpus_filepath,
        bow_generator(abstract_path)
        )
    
    trigram_bow_corpus = MmCorpus(bow_corpus_filepath)
    
    
    tfidf = models.TfidfModel(trigram_bow_corpus)
    corpus_tfidf = tfidf[trigram_bow_corpus]
    
    for doc in corpus_tfidf:
        pprint(doc)
        break
        
    lda_bow_model = gensim.models.LdaMulticore(trigram_bow_corpus, num_topics=30, id2word=dictionary_trigrams, passes=2, workers=2)
    
    print('LDA BoW MODEL')
    for idx, topic in lda_bow_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
        
    lda_tfidf_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=30, id2word=dictionary_trigrams, passes=2, workers=2)
    
    print('LDA TFIDF MODEL')
    for idx, topic in lda_tfidf_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
        
    return lda_bow_model,lda_tfidf_model
    
    
        
    