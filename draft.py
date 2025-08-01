import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

documents = []
vocabulary = []
lemmatizer = nltk.WordNetLemmatizer()


with open('intents.json', 'r') as f:
    data = json.load(f)


for intent in data['intents']:
    for pattern in intent['patterns']:
        
        tokens = word_tokenize(pattern.lower())
        lemmas = [lemmatizer.lemmatize(word) for word in tokens]

        vocabulary.extend(lemmas)
        documents.append((lemmas, intent['tag']))

for d in documents[:3]:
    print(d)
for word in vocabulary[:10]:
    print(word)

