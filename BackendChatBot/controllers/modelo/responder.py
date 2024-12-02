import nltk
import pickle
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from data_crud.models import Intent  # Asumiendo que tienes un modelo 'Intent' en tu base de datos

class Chatbot:
    def __init__(self):
        # Cargar el modelo
        self.model = load_model('BackendChatBot/controllers/modelo/chatbot_model.keras')
        
        # Cargar las palabras y las clases
        self.words = pickle.load(open('BackendChatBot/controllers/modelo/words.pkl', 'rb'))
        self.classes = pickle.load(open('BackendChatBot/controllers/modelo/classes.pkl', 'rb'))
        
        # Inicializar el lematizador
        self.lemmatizer = WordNetLemmatizer()
        
        # Diccionario para almacenar respuestas de los intents (carga diferida)
        self.responses_cache = {}

    def load_intent_responses(self, tag):
        """Carga las respuestas de un intent específico desde la base de datos, si no están cacheadas."""
        if tag not in self.responses_cache:
            # Buscar el Intent por tag en la base de datos
            try:
                intent = Intent.objects.get(tag=tag)
                # Guardar las respuestas en el cache
                self.responses_cache[tag] = [response.text for response in intent.responses.all()]
            except Intent.DoesNotExist:
                self.responses_cache[tag] = []  # Si no existe el intent, no hay respuestas

        return self.responses_cache[tag]

    # Función para limpiar la entrada
    def clean_up_sentence(self, sentence):
        # Tokenizar la oración
        sentence_words = nltk.word_tokenize(sentence)
        # Lematizar las palabras
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    
    # Función para convertir la oración en un vector de palabras
    def bow(self, sentence, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print(f"found in bag: {w}")
        return np.array(bag)
    
    # Función para predecir la clase de la oración
    def predict_class(self, sentence):
        p = self.bow(sentence, show_details=False)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = [{"intent": self.classes[r[0]], "probability": str(r[1])} for r in results]
        return return_list
    
    # Función para obtener la respuesta a partir de la clase predicha
    def get_response(self, ints):
        tag = ints[0]['intent']
        # Cargar las respuestas desde la base de datos o cache
        responses = self.load_intent_responses(tag)
        if responses:
            return random.choice(responses)
        else:
            return "Lo siento, no tengo una respuesta para esa pregunta."

    # Función principal para preguntar y obtener respuesta
    def preguntar(self, question):
        ints = self.predict_class(question)
        response = self.get_response(ints)
        return response



# Ejemplo de uso
if __name__ == "__main__":
    chatbot = Chatbot()
    
    while True:
        question = input("Tú: ")
        if question.lower() == "salir":
            print("Bot: ¡Hasta luego!")
            break
        response = chatbot.preguntar(question)
        print(f"Bot: {response}")
