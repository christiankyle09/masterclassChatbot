import json
import nltk
import numpy as np
from sklearn.utils import shuffle
from nltk.stem.porter import PorterStemmer

#nltk.download('punkt')

# Importer le stemmer NLTK
stemmer = PorterStemmer()


# Définir la fonction de tokenization
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Définir la fonction de stemming
def stem(word):
    return stemmer.stem(word=word.lower())

# Définir la fonction du sac de mots
def bag_of_words(tokenized_pattern, vocabulary):
    tokenized_pattern = [stem(word) for word in tokenized_pattern]
    bag = np.zeros(len(vocabulary), dtype=np.float32)

    # Pour chaque mot dans le motif, nous créons un vecteur binaire
    for index, word in enumerate(vocabulary):
        if word in tokenized_pattern:
            bag[index] = 1.0

    return bag

def create_chatbot_vocabulary(file_path):
    # Lire le fichier d'intention
    with open('masterclassChatbot/intent.json', 'r') as file:
        intents = json.load(file)

    all_words = []  # Liste vide pour stocker toutes les phrases tokenisées
    tags = []  # Liste vide pour stocker les tags d'intentions
    xy = []  # Liste vide pour stocker les mots tokenisés et leurs tags correspondants sous forme de tuple

    # Boucler à travers les listes d'intentions
    for intent in intents['intents']:
        # Stocker le tag d'intention dans la liste des tags
        tags.append(intent['tag'])
        # Boucler à travers les motifs d'intention
        for pattern in intent['patterns']:
            # Appliquer la tokenisation sur le motif sous-jacent
            words = tokenize(pattern)
            # Ajouter les mots tokenisés à la liste de mots/tokens
            all_words.extend(words)
            # Stocker le tag et leurs tags correspondants
            xy.append((tags[-1], words))
    return sorted(all_words), sorted(tags), xy


def clean_chatbot_vocab(vocabulary):
    # Supprimer la ponctuation de la liste de mots ou de tokens
    ignore_list = ['?', '!', '.', ',']
    return sorted(set([stem(word) for word in vocabulary if word not in ignore_list]))


def create_train_data(vocabulary, tags, patterns):
    X_train = []
    y_train = []

    for tag, tokenized_pattern in patterns:
        # Appliquer le sac de mots sur le motif tokenisé sous-jacent
        bag = bag_of_words(tokenized_pattern=tokenized_pattern, vocabulary=vocabulary)
        # Stocker la représentation numérique de l'échantillon de motif sur l'ensemble d'entraînement
        X_train.append(bag)
        # Stocker l'étiquette de tag correspondante sur l'ensemble d'entraînement
        y_train.append(tags.index(tag))
    return np.array(X_train), np.array(y_train)


# Créer le vocabulaire du chatbot
all_words, tags, tokenized_patterns = create_chatbot_vocabulary(file_path="masterclassChatbot/intent.json")

# Nettoyer le vocabulaire du chatbot
vocabulary = clean_chatbot_vocab(vocabulary=all_words)

# Créer l'ensemble d'entraînement
X_train, y_train = create_train_data(
    vocabulary=vocabulary,
    tags=tags,
    patterns=tokenized_patterns
)

X_train, y_train = shuffle(X_train, y_train, random_state=42)
