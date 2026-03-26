from django.core.management.base import BaseCommand
from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv
from movie.models import Movie

# Cargar la API Key
def get_openai_client():
    load_dotenv('./openAI.env')
    return OpenAI(api_key=os.environ.get('openai_apikey'))

# Función para calcular similitud de coseno
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class Command(BaseCommand):
    help = 'Mostrar una película aleatoria, su descripción y películas similares'

    def handle(self, *args, **options):
        client = get_openai_client()

        random_movie = Movie.objects.order_by('?').first()
        if not random_movie:
            self.stdout.write('No hay películas en la base de datos.')
            return

        self.stdout.write('--- Película aleatoria seleccionada ---')
        self.stdout.write(f'Título: {random_movie.title}')
        self.stdout.write(f'Descripción: {random_movie.description or "(sin descripción)"}')
        self.stdout.write('--------------------------------------')

        # Calcular similitudes con otras películas
        try:
            movie_emb = np.frombuffer(random_movie.emb, dtype=np.float32)
        except Exception:
            self.stdout.write('La película seleccionada no tiene embedding válido.')
            return

        similarities = []
        for movie in Movie.objects.exclude(id=random_movie.id):
            try:
                emb = np.frombuffer(movie.emb, dtype=np.float32)
            except Exception:
                continue
            sim = cosine_similarity(movie_emb, emb)
            similarities.append((sim, movie.title))

        if not similarities:
            self.stdout.write('No hay otras películas con embeddings para comparar.')
            return

        similarities.sort(key=lambda x: x[0], reverse=True)

        self.stdout.write('\n--- Películas más similares ---')
        for i, (sim, title) in enumerate(similarities[:5], start=1):
            self.stdout.write(f'{i}. {title} -> similitud {sim:.4f}')

        self.stdout.write('\n--- Películas menos similares ---')
        for i, (sim, title) in enumerate(similarities[-5:], start=1):
            self.stdout.write(f'{i}. {title} -> similitud {sim:.4f}')

