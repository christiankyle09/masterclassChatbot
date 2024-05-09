# Importer les packages requis
import streamlit as st
from keras import layers
from data_preprocessing import X_train, y_train
import json
import random
import tensorflow as tf
import numpy as np
from data_preprocessing import vocabulary, tags, tokenize, bag_of_words


# Création du cerveau du chatbot : Il s'agit d'un réseau de neurones artificiels de 3 couches
taille_entree = X_train.shape[1]
taille_sortie = np.unique(y_train).shape[0]
chatbot_brain = tf.keras.Sequential(
    [
        layers.Dense(100, input_shape=(taille_entree,), activation='relu', name='couche1'),
        layers.Dense(50, activation='relu', name='couche2'),
        layers.Dense(taille_sortie, activation='sigmoid', name='couche3')
    ]
)
chatbot_brain.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.008)
)

# Entraînement du cerveau du chatbot
historique = chatbot_brain.fit(
    X_train,
    y_train,
    epochs=70,
    batch_size=64,
    verbose=1,
)

# Ouvrir le fichier d'intention
with open("intent.json", 'r') as json_data:
    intents = json.load(json_data)

# Définition de la fonction de prédiction du modèle
def generate_chatbot_response(chatbot, user_request, tags):
    # Prédiction de l'intention de l'utilisateur
    y_pred = chatbot.predict(user_request, verbose=0)
    intent_class_id = np.argmax(y_pred)
    pred_proba = np.max(y_pred)
    intent_tag = tags[intent_class_id]
    return pred_proba, intent_tag

# Créer l'affichage du chatbot
nom_du_bot = "😊 Amine "
st.write("😊 Amine : Bonjour je suis Amine. Je veux vous aider \n\n   (tapez 'bye' pour me dire au revoir)\n")

# Initialiser une variable de compteur
compteur = 0

while True:
    # Utiliser le compteur pour générer une clé unique
    key = f"text_input_{compteur}"

    try:
        # Demander à l'utilisateur de taper sa requête
        sentence = st.text_input("👉🏼 Vous : ", key=key)

        if sentence != "":
            # Vérifier si l'utilisateur souhaite mettre fin à la conversation
            if sentence.lower() in ["bye", "ok bye", "aurevoir", "au revoir", "ok a plus", "aplus", "ok au revoir"]:
                st.write("😊 Amine : OK, Au revoir ! 👋🏻")
                break

            # Prétraiter la demande de l'utilisateur pour la rendre compréhensible par le modèle d'IA
            sentence = tokenize(sentence)
            X = bag_of_words(sentence, vocabulary=vocabulary)
            X = X.reshape(1, -1)

            # Prédire l'intention de l'utilisateur
            pred_proba, intent_tag = generate_chatbot_response(chatbot_brain, X, tags)

            # Si le chatbot est sûr d'avoir compris la demande de l'utilisateur
            # Vérifie si la probabilité prédite est supérieure à 0.75
            if pred_proba > 0.75:
                # Parcours chaque intention dans la liste d'intentions
                for intent in intents['intents']:
                    # Vérifie si l'étiquette de l'intention correspond à l'intention actuelle
                    if intent_tag == intent['tag']:
                        # Affiche une réponse aléatoire associée à l'intention
                        st.write(f"{nom_du_bot}: {random.choice(intent['responses'])}")
                        compteur += 1

            # Sinon
            else:
                st.write(f"{nom_du_bot}: \
                Désolé, je ne comprends pas... Pourriez-vous reformuler votre demande s'il vous plaît ?")

    except st.errors.DuplicateWidgetID as e:
        pass
