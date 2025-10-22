import logging
import cv2
import numpy as np
from fer.fer import FER
import os

# Configurar logging
logger = logging.getLogger("app.emotions")

# Inicializar el detector de emociones (usa MTCNN para detección de rostros)
detector = FER(mtcnn=True)

# Mapeo de emociones en inglés a español
MAPEO_EMOCIONES = {
    'angry': 'Enojo',
    'disgust': 'Disgusto', 
    'fear': 'Miedo',
    'happy': 'Felicidad',
    'sad': 'Tristeza',
    'surprise': 'Sorpresa',
    'neutral': 'Neutral'
}

class AnalizadorEmocionesFER:
    def __init__(self):
        self.detector = detector
        self.emociones = list(MAPEO_EMOCIONES.values())
        logger.info("✅ Analizador FER inicializado correctamente")
    
    def detect_emotions_for_face(self, face_image):
        """
        Detecta la emoción predominante en una cara usando FER.
        Retorna (emotion_name, confidence) o ("Neutral", 0.0) si falla.
        """
        try:
            # Verificar que la imagen tiene tamaño adecuado
            if face_image is None or face_image.size == 0:
                return "Neutral", 0.0
            
            # Asegurar que la imagen tiene el tamaño mínimo requerido
            if face_image.shape[0] < 20 or face_image.shape[1] < 20:
                # Redimensionar a tamaño mínimo
                face_image = cv2.resize(face_image, (48, 48))
            
            # Convertir BGR a RGB si es necesario (FER espera RGB)
            if len(face_image.shape) == 3:
                # Verificar el orden de canales
                if face_image.shape[2] == 3:
                    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                else:
                    face_image_rgb = face_image
            else:
                # Convertir escala de grises a RGB
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            
            # Detectar emoción principal
            emotion_data = self.detector.detect_emotions(face_image_rgb)
            
            if not emotion_data:
                return "Neutral", 0.0
            
            # Tomar el primer rostro detectado
            emotions = emotion_data[0]['emotions']
            
            # Encontrar la emoción con mayor confianza
            emotion_english = max(emotions.items(), key=lambda x: x[1])[0]
            confidence = emotions[emotion_english]
            
            # Traducir al español
            emotion_spanish = MAPEO_EMOCIONES.get(emotion_english, "Neutral")
            
            logger.debug(f"Emoción detectada: {emotion_english} -> {emotion_spanish} ({confidence:.3f})")
            return emotion_spanish, float(confidence)

        except Exception as e:
            logger.error(f"Error detectando emoción con FER: {e}", exc_info=True)
            return "Neutral", 0.0
    
    def predecir_emocion(self, frame, ubicacion_rostro):
        """
        Predice la emoción de un rostro en el frame usando FER.
        
        Args:
            frame: Frame completo de la cámara
            ubicacion_rostro: Tupla (top, right, bottom, left) con la ubicación del rostro
        
        Returns:
            tuple: (emoción, confianza)
        """
        try:
            # Extraer región del rostro
            top, right, bottom, left = ubicacion_rostro
            
            # Asegurar coordenadas válidas
            top = max(0, top)
            left = max(0, left)
            bottom = min(frame.shape[0], bottom)
            right = min(frame.shape[1], right)
            
            if top >= bottom or left >= right:
                return "Neutral", 0.0
                
            rostro_region = frame[top:bottom, left:right]
            
            if rostro_region.size == 0:
                return "Neutral", 0.0
            
            # Usar FER para detectar emociones
            return self.detect_emotions_for_face(rostro_region)
            
        except Exception as e:
            logger.error(f"Error en predicción de emoción: {e}")
            return "Neutral", 0.0
    
    def detectar_rostros(self, frame):
        """
        Detecta rostros en el frame usando el detector interno de FER.
        Esto puede ser útil como alternativa al detector de face_recognition.
        """
        try:
            # Convertir a RGB para FER
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            # Detectar rostros con FER
            emotion_data = self.detector.detect_emotions(frame_rgb)
            
            ubicaciones = []
            for detection in emotion_data:
                box = detection['box']  # [x, y, width, height]
                x, y, w, h = box
                # Convertir a formato (top, right, bottom, left)
                top = y
                right = x + w
                bottom = y + h
                left = x
                ubicaciones.append((top, right, bottom, left))
            
            return ubicaciones
            
        except Exception as e:
            logger.error(f"Error detectando rostros con FER: {e}")
            return []

# Instancia global para usar en el sistema
analizador_emociones = AnalizadorEmocionesFER()