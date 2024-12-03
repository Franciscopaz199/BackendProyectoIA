# BackendChatBot/management/commands/entrenar.py

from django.core.management.base import BaseCommand
from BackendChatBot.controllers.modelo.entrenar import ChatbotTrainer

class Command(BaseCommand):
    help = 'Entrena el modelo con los datos de intentos'

    def handle(self, *args, **kwargs):
        chatbot_trainer = ChatbotTrainer()
        chatbot_trainer.run()
        # Aquí va la lógica de tu script de entrenamiento.
        print("Entrenando el modelo")

