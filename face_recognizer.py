import face_recognition
import cv2
import numpy as np
from database import DatabaseManager
import logging

logger = logging.getLogger(__name__)

class ReconocedorFacial:
    def __init__(self, db_manager):
        self.db = db_manager
        self.tolerancia_reconocimiento = 0.6
        self.capturas_por_registro = 3
        self.capturas_realizadas = 0
        self.embeddings_registro = []
        self.cache_personas = None
        self.cache_embeddings = None
        self.actualizar_cache()
    
    def extraer_embedding_rostro(self, frame):
        """Extrae el embedding facial de un frame - OPTIMIZADO"""
        try:
            # Reducir tamaño del frame para mayor velocidad
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            # Convertir BGR a RGB
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detectar ubicaciones de rostros
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            
            if not face_locations:
                return None, None
            
            # Extraer embeddings
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            if not face_encodings:
                return None, None
            
            # Escalar ubicaciones de vuelta al tamaño original
            top, right, bottom, left = face_locations[0]
            top *= 2; right *= 2; bottom *= 2; left *= 2
            
            return face_encodings[0], (top, right, bottom, left)
            
        except Exception as e:
            logger.error(f"Error al extraer embedding: {str(e)}")
            return None, None
    
    def capturar_para_registro(self, frame):
        """Captura múltiples imágenes para registro"""
        embedding, ubicacion = self.extraer_embedding_rostro(frame)
        
        if embedding is not None:
            self.embeddings_registro.append(embedding)
            self.capturas_realizadas += 1
            return True, ubicacion
        else:
            return False, None
    
    def finalizar_registro(self):
        """Finaliza el registro promediando los embeddings capturados"""
        if len(self.embeddings_registro) < self.capturas_por_registro:
            return None
        
        # Promediar los embeddings para mayor robustez
        embedding_promedio = np.mean(self.embeddings_registro, axis=0)
        
        # Reiniciar contadores
        self.capturas_realizadas = 0
        self.embeddings_registro = []
        
        # Actualizar cache después de nuevo registro
        self.actualizar_cache()
        
        return embedding_promedio
    
    def reiniciar_registro(self):
        """Reinicia el proceso de registro"""
        self.capturas_realizadas = 0
        self.embeddings_registro = []
    
    def reconocer_persona(self, frame):
        """Reconoce una persona en el frame - OPTIMIZADO CON CACHE"""
        embedding, ubicacion = self.extraer_embedding_rostro(frame)
        
        if embedding is None:
            return None, None, ubicacion
        
        # Usar cache para reconocimiento más rápido
        if self.cache_embeddings and self.cache_personas:
            face_distances = face_recognition.face_distance(self.cache_embeddings, embedding)
            best_match_index = np.argmin(face_distances)
            
            if face_distances[best_match_index] < self.tolerancia_reconocimiento:
                mejor_coincidencia = self.cache_personas[best_match_index]
                mejor_distancia = face_distances[best_match_index]
                
                # Convertir distancia a confianza (0-100%)
                confianza = max(0, min(100, (1 - mejor_distancia) * 100))
                return mejor_coincidencia, confianza, ubicacion
        
        return None, None, ubicacion
    
    def actualizar_cache(self):
        """Actualiza la cache de personas y embeddings para reconocimiento más rápido"""
        try:
            personas = self.db.obtener_todas_personas()
            self.cache_personas = personas
            self.cache_embeddings = []
            
            for persona in personas:
                embedding = self.db.obtener_embedding_persona(persona.id)
                if embedding is not None:
                    self.cache_embeddings.append(embedding)
            
            logger.info(f"Cache actualizado: {len(personas)} personas cargadas")
        except Exception as e:
            logger.error(f"Error al actualizar cache: {str(e)}")