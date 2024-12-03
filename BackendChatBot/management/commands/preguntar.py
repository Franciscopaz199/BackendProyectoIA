# BackendChatBot/management/commands/entrenar.py

from django.core.management.base import BaseCommand
from BackendChatBot.controllers.modelo.responder import Chatbot

class Command(BaseCommand):
    help = 'Entrena el modelo con los datos de intentos'

    def handle(self, *args, **kwargs):
        chatbot = Chatbot()
        while True:
            question = input("Tú: ")
            if question.lower() == "salir":
                print("Bot: ¡Hasta luego!")
                break
            response = chatbot.preguntar(question)
            print(f"Bot: {response}")



