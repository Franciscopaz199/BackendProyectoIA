FROM python:3.10.6

ENV PYTHONUNBUFFERED 1
RUN mkdir /code

WORKDIR /code
COPY . /code/

# Instalar las dependencias de Python
RUN pip install -r requirements.txt

# Descargar los recursos de NLTK necesarios (en este caso 'punkt_tab' y 'wordnet')
RUN python -m nltk.downloader punkt_tab
RUN python -m nltk.downloader wordnet

# Configurar y ejecutar la aplicación con Gunicorn
CMD ["gunicorn", "-c", "config/gunicorn/conf.py", "--bind", ":8000", "BackendChatBot.wsgi:application"]
