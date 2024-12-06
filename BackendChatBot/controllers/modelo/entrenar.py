import nltk
import json
import pickle
import numpy as np
import random
import os
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from data_crud.models import Intent


class ChatbotTrainer:
    def __init__(self, intents_file=None):
        self.intents_file = intents_file
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!']
        self.intents = None
        self.train_x = []
        self.train_y = []

    # Cargar intenciones desde el archivo JSON
    def load_intents(self):
        with open(self.intents_file, 'r') as file:
            self.intents = json.load(file)

    # Cargar intenciones desde la base de datos
    def train_by_database(self):
        self.intents = []

        # Usamos prefetch_related para evitar N+1 consultas
        intents = Intent.objects.prefetch_related('patterns').all()

        for intent in intents:
            intent_data = {'tag': intent.tag, 'patterns': [], 'responses': [response.text for response in intent.responses.all()]}
            
            # Acceder a los patrones prefetchados
            for pattern in intent.patterns.all():
                intent_data['patterns'].append(pattern.text)

            self.intents.append(intent_data)

        print(self.intents)

    def preprocess_data(self):
        for intent in self.intents:
            for pattern in intent['patterns']:
                # Tokenize each word
                tokens = nltk.word_tokenize(pattern)
                self.words.extend(tokens)
                # Add to documents
                self.documents.append((tokens, intent['tag']))
                # Add to classes if not already present
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # Lemmatize and sort words and classes
        self.words = sorted(set(self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words))
        self.classes = sorted(set(self.classes))

        # Save words and classes for future use
        pickle.dump(self.words, open('BackendChatBot/controllers/modelo/words.pkl', 'wb'))
        pickle.dump(self.classes, open('BackendChatBot/controllers/modelo/classes.pkl', 'wb'))

    def create_training_data(self):
        training = []
        output_empty = [0] * len(self.classes)

        for doc in self.documents:
            bag = []
            pattern_words = [self.lemmatizer.lemmatize(w.lower()) for w in doc[0]]
            bag = [1 if w in pattern_words else 0 for w in self.words]

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1

            training.append([bag, output_row])

        random.shuffle(training)
        self.train_x = np.array([item[0] for item in training])
        self.train_y = np.array([item[1] for item in training])

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(len(self.train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.train_y[0]), activation='softmax'))

        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def train_model(self, model, epochs=300, batch_size=5):
        model.fit(self.train_x, self.train_y, epochs=epochs, batch_size=batch_size, verbose=1)
        model.save('BackendChatBot/controllers/modelo/chatbot_model.keras')
        print("Modelo creado y guardado en 'chatbot_model.keras'")

    def run(self):
        self.train_by_database()
        self.preprocess_data()
        print(f"{len(self.documents)} documentos")
        print(f"{len(self.classes)} clases: {self.classes}")
        print(f"{len(self.words)} palabras únicas lematizadas: {self.words}")

        self.create_training_data()
        print("Datos de entrenamiento creados")

        model = self.build_model()
        self.train_model(model)

# Ejecución principal
if __name__ == "__main__":
    print("PATH: ", os.getcwd())
    trainer = ChatbotTrainer()
    trainer.run()

