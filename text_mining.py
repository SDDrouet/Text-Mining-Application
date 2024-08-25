import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
import spacy
from spacy.language import Language
from spacy_lefff import LefffLemmatizer, POSTagger
from spacytextblob.spacytextblob import SpacyTextBlob
from translate import Translator
# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('stopwords')

# Registrar los componentes personalizados
@Language.factory('lefff_lemmatizer')
def create_lefff_lemmatizer(nlp, name):
    return LefffLemmatizer()

@Language.factory('pos_tagger')
def create_pos_tagger(nlp, name):
    return POSTagger()

# Cargar el modelo de spaCy y añadir componentes
nlp = spacy.load("es_core_news_sm")
nlp.add_pipe('spacytextblob')

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def analyze_sentiment(text):
    try:
        translator = Translator(from_lang="es", to_lang='en')
        texto_traducido = translator.translate(text)
    except:
        print('Error al traducir el texto')
        texto_traducido = text

    doc = nlp(texto_traducido)
    return doc._.blob.polarity

def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    freq = FreqDist(words)
    ranking = {}
    
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                if i in ranking:
                    ranking[i] += freq[word]
                else:
                    ranking[i] = freq[word]
    
    top_sentences = sorted(ranking, key=ranking.get, reverse=True)[:num_sentences]
    summary = [sentences[i] for i in sorted(top_sentences)]
    
    return ' '.join(summary)

def analyze_text(text):
    entities = extract_entities(text)
    sentiment = analyze_sentiment(text)
    summary = summarize_text(text)
    
    return {
        'entities': entities,
        'sentiment': sentiment,
        'summary': summary
    }