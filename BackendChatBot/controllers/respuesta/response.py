
from openai import OpenAI
import google.generativeai as genai
from BackendChatBot.controllers.modelo.responder import Chatbot
import os
import json



class Response:

    prompt = '''Tengo un chatbot llamado "Pancho Bot", diseñado para responder preguntas frecuentes sobre cómo ingresar a la carrera de Ingeniería de Sistemas en la UNAH. Sin embargo, Pancho Bot no siempre da respuestas precisas, 
    por lo que necesito que evalúes las respuestas que proporciona y tomes las acciones correspondientes en los siguientes casos:
    1. **Si la respuesta tiene sentido en relación con la pregunta**: Reescribe la respuesta de forma más clara y comprensible, para que las personas puedan entenderla mejor.
    2. **Si la respuesta no tiene sentido en relación con la pregunta**: Indica que la respuesta no es adecuada y proporciona una respuesta correcta y coherente para esa pregunta.
    3. **Si la pregunta está fuera del ámbito de Ingeniería de Sistemas**: Informa que tus datos de entrenamiento van más allá de la pregunta, pero basándote en internet, da una respuesta relacionada.
    4. **Asume siempre el rol de "Pancho Bot" para responder las preguntas**: Responde como si fueras el chatbot, manteniendo el tono y el estilo adecuado para un asistente virtual, no es necesario que digas que eres pancho bot en cada respuesta, solo si te lo piden.
    Por favor, sigue estas instrucciones para mejorar la calidad de las respuestas y hacer que el chatbot sea más útil para los usuarios.'''




    @staticmethod
    def generate_response(question, modelo):
        if modelo == "gemeni":
            return Response.response_with_gemeni(question)
        elif modelo == "gpt4":
            return Response.response_with_chat_gpt(question)
        elif modelo == "panchobot":
            return Response.response_with_pancho_bot(question)
        else:
            return "Modelo no soportado"

    @staticmethod
    def response_with_gemeni(question):
        genai.configure(api_key=os.getenv("GENAI_API_KEY"))
        model = genai.GenerativeModel(os.getenv("GENAI_MODEL"))
        # CREAR UN DICCIONARIO CON LA PREGUNTA, LA RESPUESTA DE PANCHOBOT Y EL PROMPT
        data = {
            "prompt": Response.prompt,
            "question": question,
            "response_by_pancho_bot": Response.response_with_pancho_bot(question),
        }

            # Convertir el diccionario a formato JSON (texto)
        data_text = json.dumps(data)

         # Pasar el texto al modelo Gemini
        response = model.generate_content(data_text)
        return response.text + "\n\n Response by Gemeni"
        

    @staticmethod
    def response_with_chat_gpt(question):
        data = {
            "prompt": Response.prompt,
            "question": question,
            "response_by_pancho_bot": Response.response_with_pancho_bot(question),
        }

        data_text = json.dumps(data)

        try:
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            completion = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL"),
                messages=[
                    {
                        "role": "user",
                        "content": data_text,
                    },
                ],
            )

            return completion.choices[0].message.content + "\n\n Response by GPT-4"
        except Exception as e:
            return f"Error al conectarse a la API: {str(e)}"
    
    @staticmethod
    def response_with_pancho_bot(question):
        # Crear una instancia de la clase gemeni
        chatbot = Chatbot()
        # Obtener la respuesta a partir de la pregunta
        response = chatbot.preguntar(question)
        return response + "\n\n Response by PanchoBot"
     
