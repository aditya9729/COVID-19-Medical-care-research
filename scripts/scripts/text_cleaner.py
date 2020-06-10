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

    