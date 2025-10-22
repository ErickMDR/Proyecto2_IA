# Sistema de Reconocimiento Facial con Análisis de Emociones

**Desarrollado por:** Erick Díaz C.I.29963164

**Materia:** Inteligencia Artificial

**Periodo:** 2025C

## Descripción
Sistema completo de reconocimiento facial con análisis de emociones en tiempo real, desarrollado en Python con interfaz gráfica Tkinter.

## Características
- **Registro facial** con captura múltiple
- **Reconocimiento en tiempo real**
- **Análisis de 7 emociones** básicas
- **Base de datos SQLite** local
- **Reportes PDF** con estadísticas
- **Interfaz gráfica** intuitiva

## Estructura del Proyecto

facial_emotion_system/

├── main.py                 # Punto de entrada

├── database.py             # Gestión de base de datos

├── face_recognizer.py      # Reconocimiento facial

├── emotion_analyzer.py     # Análisis de emociones

├── gui.py                  # Interfaz gráfica

├── report_generator.py     # Generador de reportes PDF

└── requirements.txt        # Dependencias

## Uso del Sistema

1. Registro de Personas
- Ir a pestaña "Registro"
- Completar datos personales
- Hacer clic en "Iniciar Captura"
- Posicionarse frente a la cámara
- Registrar cuando se completen las 3 capturas

2. Detección en Tiempo Real
- Ir a pestaña "Detección"
- Hacer clic en "Iniciar Detección"
- El sistema identificará personas y emociones automáticamente

3. Generación de Reportes 
- Ir a pestaña "Reportes"
- Seleccionar persona o generar reporte general
- Exportar a PDF

## Requisitos del Sistema
- Windows 10/11
- Python 3.8+
- Cámara web: mínimo 640x480 (recomendado 720p)

## Instalación

1. Descargar el proyecto
2. Ejecutar en la carpeta del proyecto:
```bash
pip install -r requirements.txt
python main.py
