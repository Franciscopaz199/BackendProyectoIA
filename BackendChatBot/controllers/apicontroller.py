from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from BackendChatBot.controllers.respuesta.response import Response
import os

@csrf_exempt  # Exime la vista de la protección CSRF para pruebas (solo en desarrollo)
def test_view(request):
    if request.method == 'POST':
        try:
            # Cargar el cuerpo de la solicitud como JSON
            data = json.loads(request.body)

            # Obtener el valor de 'question' desde el JSON
            question = data.get('question')
            modelo = data.get('model')

            if not question or not modelo:
                return JsonResponse({'error': 'Question is required'}, status=400)
            
            # Configurar el modelo generativo con la API Key
            response,modelo = Response.generate_response(question, modelo)
            
            # Devolver la respuesta generada
            return JsonResponse({'answer': response,
                                'modelo': modelo
                                 }, status=200)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)

    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
