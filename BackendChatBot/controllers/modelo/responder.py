import nltk
import pickle
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from data_crud.models import Intent  # Asumiendo que tienes un modelo 'Intent' en tu base de datos

class Chatbot:
    def __init__(self):
        try:
            # Cargar el modelo
            self.model = load_model('BackendChatBot/controllers/modelo/chatbot_model.keras')
            
            # Cargar las palabras y las clases
            self.words = pickle.load(open('BackendChatBot/controllers/modelo/words.pkl', 'rb'))
            self.classes = pickle.load(open('BackendChatBot/controllers/modelo/classes.pkl', 'rb'))
            
            # Inicializar el lematizador
            self.lemmatizer = WordNetLemmatizer()
            
            # Diccionario para almacenar respuestas de los intents (carga diferida)
            self.responses_cache = {}
        except Exception as e:
            print("Error al inicializar el chatbot:", e)

    def load_intent_responses(self, tag):
        """Carga las respuestas de un intent específico desde la base de datos, si no están cacheadas."""
        try:
            if tag not in self.responses_cache:
                # Buscar el Intent por tag en la base de datos
                intent = Intent.objects.get(tag=tag)
                # Guardar las respuestas en el cache
                self.responses_cache[tag] = [response.text for response in intent.responses.all()]

            return self.responses_cache[tag]
        except Intent.DoesNotExist:
            return []  # Si no existe el intent, no hay respuestas
        except Exception as e:
            print("Error al cargar las respuestas del intent:", e)
            return []

    def clean_up_sentence(self, sentence):
        """Limpia y tokeniza la oración de entrada."""
        try:
            sentence_words = nltk.word_tokenize(sentence)
            sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
            return sentence_words
        except Exception as e:
            print("Error al limpiar la oración:", e)
            return []

    def bow(self, sentence, show_details=True):
        """Convierte la oración en un vector de palabras (bag of words)."""
        try:
            sentence_words = self.clean_up_sentence(sentence)
            bag = [0] * len(self.words)
            
            for s in sentence_words:
                for i, w in enumerate(self.words):
                    if w == s:
                        bag[i] = 1
                        if show_details:
                            print(f"found in bag: {w}")
            return np.array(bag)
        except Exception as e:
            print("Error al crear el vector de palabras:", e)
            return np.zeros(len(self.words))

    def predict_class(self, sentence):
        """Predice la clase de la oración."""
        try:
            p = self.bow(sentence, show_details=False)
            res = self.model.predict(np.array([p]))[0]
            ERROR_THRESHOLD = 0.25
            results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
            
            results.sort(key=lambda x: x[1], reverse=True)
            return_list = [{"intent": self.classes[r[0]], "probability": str(r[1])} for r in results]
            return return_list
        except Exception as e:
            print("Error al predecir la clase:", e)
            return []

    def get_response(self, ints):
        """Obtiene una respuesta basada en la clase predicha."""
        try:
            if not ints:
                return "Lo siento, no pude entender tu pregunta."
            tag = ints[0]['intent']
            responses = self.load_intent_responses(tag)
            if responses:
                return random.choice(responses)
            else:
                return "Lo siento, no tengo una respuesta para esa pregunta."
        except Exception as e:
            print("Error al obtener la respuesta:", e)
            return "Lo siento, no pude entender tu pregunta."

    def preguntar(self, question):
        """Función principal para procesar preguntas y devolver respuestas."""
        try:
            ints = self.predict_class(question)
            response = self.get_response(ints)
            return response
        except Exception as e:
            print("Error en el procesamiento de la pregunta:", e)
            return "Lo siento, no pude entender tu pregunta."

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
