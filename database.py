import sqlalchemy as db
from sqlalchemy import Column, Integer, String, DateTime, Float, LargeBinary, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timedelta
import numpy as np
import pickle

Base = declarative_base()

class Persona(Base):
    __tablename__ = 'personas'
    
    id = Column(Integer, primary_key=True)
    nombre = Column(String(100), nullable=False)
    apellido = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    fecha_registro = Column(DateTime, default=datetime.now)
    embedding_facial = Column(LargeBinary, nullable=False)
    detecciones = relationship("DeteccionEmocion", back_populates="persona")

class DeteccionEmocion(Base):
    __tablename__ = 'detecciones_emociones'
    
    id = Column(Integer, primary_key=True)
    persona_id = Column(Integer, ForeignKey('personas.id'))
    emocion = Column(String(50), nullable=False)
    confianza = Column(Float, nullable=False)
    fecha_deteccion = Column(DateTime, default=datetime.now)
    persona = relationship("Persona", back_populates="detecciones")

class DatabaseManager:
    def __init__(self, db_path='facial_emotion_system.db'):
        self.engine = db.create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def registrar_persona(self, nombre, apellido, email, embedding):
        """Registra una nueva persona en la base de datos"""
        try:
            # Verificar si el email ya existe
            if self.session.query(Persona).filter_by(email=email).first():
                return False, "El email ya está registrado"
            
            # Serializar el embedding (array numpy) a bytes
            embedding_bytes = pickle.dumps(embedding)
            
            nueva_persona = Persona(
                nombre=nombre,
                apellido=apellido,
                email=email,
                embedding_facial=embedding_bytes
            )
            
            self.session.add(nueva_persona)
            self.session.commit()
            return True, "Persona registrada exitosamente"
            
        except Exception as e:
            self.session.rollback()
            return False, f"Error al registrar persona: {str(e)}"
    
    def buscar_persona_por_email(self, email):
        """Busca una persona por su email"""
        return self.session.query(Persona).filter_by(email=email).first()
    
    def obtener_todas_personas(self):
        """Obtiene todas las personas registradas"""
        return self.session.query(Persona).all()
    
    def obtener_persona_por_id(self, persona_id):
        """Obtiene una persona por su ID"""
        return self.session.query(Persona).filter_by(id=persona_id).first()
    
    def obtener_embedding_persona(self, persona_id):
        """Obtiene y deserializa el embedding de una persona"""
        persona = self.session.query(Persona).filter_by(id=persona_id).first()
        if persona and persona.embedding_facial:
            return pickle.loads(persona.embedding_facial)
        return None
    
    def registrar_deteccion(self, persona_id, emocion, confianza):
        """Registra una detección de emoción en el historial"""
        try:
            nueva_deteccion = DeteccionEmocion(
                persona_id=persona_id,
                emocion=emocion,
                confianza=confianza
            )
            self.session.add(nueva_deteccion)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            print(f"Error al registrar detección: {e}")
            return False
    
    def obtener_historial_emociones(self, persona_id=None, dias=30):
        """Obtiene el historial de emociones, filtrado por persona y días"""
        query = self.session.query(DeteccionEmocion)
        
        if persona_id:
            query = query.filter_by(persona_id=persona_id)
        
        fecha_limite = datetime.now() - timedelta(days=dias)
        query = query.filter(DeteccionEmocion.fecha_deteccion >= fecha_limite)
        
        return query.all()
    
    def obtener_estadisticas_emociones(self, persona_id=None):
        """Obtiene estadísticas de emociones por persona o generales"""
        if persona_id:
            detecciones = self.session.query(DeteccionEmocion).filter_by(persona_id=persona_id).all()
        else:
            detecciones = self.session.query(DeteccionEmocion).all()
        
        estadisticas = {}
        for deteccion in detecciones:
            if deteccion.emocion not in estadisticas:
                estadisticas[deteccion.emocion] = 0
            estadisticas[deteccion.emocion] += 1
        
        return estadisticas