import cv2
import tkinter as tk
from gui import SistemaReconocimientoFacial
import logging
import sys

# Configurar logging solo para nuestra aplicación
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('facial_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configurar niveles de log específicos para librerías externas
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('face_recognition').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('fer').setLevel(logging.INFO)

def main():
    try:
        print("Iniciando Sistema de Reconocimiento Facial con FER...")
        root = tk.Tk()
        app = SistemaReconocimientoFacial(root)       
        print("Sistema FER listo. Iniciando interfaz...")
        root.mainloop()
        
    except Exception as e:
        logging.error(f"Error en la aplicación: {str(e)}")
        print(f"Error crítico: {e}")

if __name__ == "__main__":
    main()