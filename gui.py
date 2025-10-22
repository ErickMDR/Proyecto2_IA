import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
import logging
from database import DatabaseManager, Persona
from face_recognizer import ReconocedorFacial
from emotion_analyzer import analizador_emociones
from report_generator import GeneradorReportes

# Configurar logging
logger = logging.getLogger(__name__)

class SistemaReconocimientoFacial:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Reconocimiento Facial con FER - Análisis de Emociones")
        self.root.geometry("1200x700")
        
        # Inicializar componentes del sistema
        self.db = DatabaseManager()
        self.reconocedor = ReconocedorFacial(self.db)
        self.analizador = analizador_emociones  # Usar FER en lugar del analizador anterior
        self.generador_reportes = GeneradorReportes(self.db)
        
        # Variables de estado optimizadas
        self.capturando = False
        self.detectando = False
        self.cap = None
        self.procesamiento_activo = False
        self.ultimo_frame = None
        self.frame_count = 0
        
        # Historial para suavizado de emociones
        self.historial_emociones = []
        self.max_historial = 5
        
        # Configurar interfaz
        self.configurar_interfaz()
        
        # Inicializar cámara en hilo separado
        self.inicializar_camara_async()
    
    def inicializar_camara_async(self):
        """Inicializa la cámara en un hilo separado"""
        def init_camera():
            try:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reducir resolución para mejor rendimiento
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                if not self.cap.isOpened():
                    self.root.after(0, lambda: messagebox.showerror("Error", "No se pudo acceder a la cámara"))
                    return
                
                logger.info("Cámara inicializada exitosamente")
                # Probar la cámara
                ret, frame = self.cap.read()
                if ret:
                    self.procesamiento_activo = True
                    self.actualizar_vista_general()
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error al inicializar cámara: {str(e)}"))
        
        # Ejecutar en hilo separado
        camera_thread = threading.Thread(target=init_camera, daemon=True)
        camera_thread.start()
    
    def configurar_interfaz(self):
        """Configura la interfaz gráfica principal"""
        # Notebook (pestañas)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Pestañas
        self.crear_pestana_registro()
        self.crear_pestana_deteccion()
        self.crear_pestana_reportes()
    
    def crear_pestana_registro(self):
        """Crea la pestaña de registro de personas"""
        self.frame_registro = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_registro, text="Registro")
        
        # Formulario de registro
        frame_formulario = ttk.LabelFrame(self.frame_registro, text="Datos Personales", padding=10)
        frame_formulario.grid(row=0, column=0, padx=10, pady=10, sticky='nw')
        
        # Campos del formulario
        ttk.Label(frame_formulario, text="Nombre:").grid(row=0, column=0, sticky='w', pady=5)
        self.entry_nombre = ttk.Entry(frame_formulario, width=30)
        self.entry_nombre.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame_formulario, text="Apellido:").grid(row=1, column=0, sticky='w', pady=5)
        self.entry_apellido = ttk.Entry(frame_formulario, width=30)
        self.entry_apellido.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(frame_formulario, text="Email:").grid(row=2, column=0, sticky='w', pady=5)
        self.entry_email = ttk.Entry(frame_formulario, width=30)
        self.entry_email.grid(row=2, column=1, padx=5, pady=5)
        
        # Botones
        frame_botones = ttk.Frame(frame_formulario)
        frame_botones.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.btn_iniciar_captura = ttk.Button(
            frame_botones, 
            text="Iniciar Captura", 
            command=self.iniciar_captura_registro
        )
        self.btn_iniciar_captura.pack(side='left', padx=5)
        
        self.btn_detener_captura = ttk.Button(
            frame_botones, 
            text="Detener Captura", 
            command=self.detener_captura_registro,
            state='disabled'
        )
        self.btn_detener_captura.pack(side='left', padx=5)
        
        self.btn_registrar = ttk.Button(
            frame_botones, 
            text="Registrar Persona", 
            command=self.registrar_persona,
            state='disabled'
        )
        self.btn_registrar.pack(side='left', padx=5)
        
        # Vista previa de cámara
        frame_camara = ttk.LabelFrame(self.frame_registro, text="Vista Previa", padding=10)
        frame_camara.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky='nsew')
        
        self.label_camara = ttk.Label(frame_camara)
        self.label_camara.pack()
        
        # Indicador de progreso
        self.progress_registro = ttk.Progressbar(
            frame_camara, 
            orient='horizontal', 
            length=300, 
            mode='determinate',
            maximum=self.reconocedor.capturas_por_registro
        )
        self.progress_registro.pack(pady=10)
        
        self.label_progreso = ttk.Label(frame_camara, text="Capturas: 0/3")
        self.label_progreso.pack()
        
        # Mensajes de estado
        self.label_estado_registro = ttk.Label(frame_camara, text="Listo para capturar")
        self.label_estado_registro.pack(pady=5)
        
        # Configurar grid weights
        self.frame_registro.columnconfigure(1, weight=1)
        self.frame_registro.rowconfigure(0, weight=1)
    
    def crear_pestana_deteccion(self):
        """Crea la pestaña de detección en tiempo real"""
        self.frame_deteccion = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_deteccion, text="Detección en Tiempo Real")
        
        # Vista de cámara
        frame_camara_deteccion = ttk.LabelFrame(self.frame_deteccion, text="Detección en Tiempo Real", padding=10)
        frame_camara_deteccion.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.label_camara_deteccion = ttk.Label(frame_camara_deteccion)
        self.label_camara_deteccion.pack()
        
        # Información de detección
        frame_info = ttk.Frame(frame_camara_deteccion)
        frame_info.pack(fill='x', pady=10)
        
        self.label_info_persona = ttk.Label(frame_info, text="Persona: No detectada", font=('Arial', 12))
        self.label_info_persona.pack()
        
        self.label_info_emocion = ttk.Label(frame_info, text="Emoción: -", font=('Arial', 12))
        self.label_info_emocion.pack()
        
        self.label_info_confianza = ttk.Label(frame_info, text="Confianza: -", font=('Arial', 12))
        self.label_info_confianza.pack()
        
        # Controles
        frame_controles = ttk.Frame(frame_camara_deteccion)
        frame_controles.pack(fill='x', pady=10)
        
        self.btn_iniciar_deteccion = ttk.Button(
            frame_controles, 
            text="Iniciar Detección", 
            command=self.iniciar_deteccion
        )
        self.btn_iniciar_deteccion.pack(side='left', padx=5)
        
        self.btn_detener_deteccion = ttk.Button(
            frame_controles, 
            text="Detener Detección", 
            command=self.detener_deteccion,
            state='disabled'
        )
        self.btn_detener_deteccion.pack(side='left', padx=5)
        
        self.btn_actualizar_cache = ttk.Button(
            frame_controles,
            text="Actualizar Cache",
            command=self.actualizar_cache_sistema
        )
        self.btn_actualizar_cache.pack(side='left', padx=5)
    
    def crear_pestana_reportes(self):
        """Crea la pestaña de reportes y estadísticas"""
        self.frame_reportes = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_reportes, text="Reportes")
        
        # Selección de persona
        frame_seleccion = ttk.LabelFrame(self.frame_reportes, text="Generar Reporte", padding=10)
        frame_seleccion.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(frame_seleccion, text="Persona:").grid(row=0, column=0, padx=5, pady=5)
        
        self.combo_personas = ttk.Combobox(frame_seleccion, state="readonly", width=30)
        self.combo_personas.grid(row=0, column=1, padx=5, pady=5)
        
        self.btn_actualizar_lista = ttk.Button(
            frame_seleccion, 
            text="Actualizar Lista", 
            command=self.actualizar_lista_personas
        )
        self.btn_actualizar_lista.grid(row=0, column=2, padx=5, pady=5)
        
        # Botones de reporte
        frame_botones_reporte = ttk.Frame(frame_seleccion)
        frame_botones_reporte.grid(row=1, column=0, columnspan=3, pady=10)
        
        self.btn_reporte_persona = ttk.Button(
            frame_botones_reporte,
            text="Generar Reporte de Persona",
            command=self.generar_reporte_persona
        )
        self.btn_reporte_persona.pack(side='left', padx=5)
        
        self.btn_reporte_general = ttk.Button(
            frame_botones_reporte,
            text="Generar Reporte General",
            command=self.generar_reporte_general
        )
        self.btn_reporte_general.pack(side='left', padx=5)
        
        # Área de visualización
        frame_visualizacion = ttk.LabelFrame(self.frame_reportes, text="Vista Previa", padding=10)
        frame_visualizacion.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.texto_reportes = tk.Text(frame_visualizacion, height=15, width=80)
        scrollbar = ttk.Scrollbar(frame_visualizacion, orient='vertical', command=self.texto_reportes.yview)
        self.texto_reportes.configure(yscrollcommand=scrollbar.set)
        
        self.texto_reportes.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Actualizar lista inicial
        self.actualizar_lista_personas()
    
    def actualizar_vista_general(self):
        """Actualiza la vista de cámara general cuando no hay procesos activos"""
        if not self.procesamiento_activo or not self.cap:
            return
        
        try:
            ret, frame = self.cap.read()
            if ret:
                # Mostrar vista simple sin procesamiento
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Actualizar todas las vistas de cámara
                for label in [self.label_camara, self.label_camara_deteccion]:
                    if hasattr(label, 'img_tk'):
                        label.img_tk = img_tk
                        label.config(image=img_tk)
                
            # Continuar actualización
            if self.procesamiento_activo:
                self.root.after(50, self.actualizar_vista_general)
                
        except Exception as e:
            logger.error(f"Error en actualización general: {str(e)}")
    
    def iniciar_captura_registro(self):
        """Inicia el proceso de captura para registro"""
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Cámara no disponible")
            return
        
        # Validar campos
        if not self.entry_nombre.get() or not self.entry_apellido.get() or not self.entry_email.get():
            messagebox.showwarning("Advertencia", "Complete todos los campos personales")
            return
        
        self.capturando = True
        self.reconocedor.reiniciar_registro()
        
        self.btn_iniciar_captura.config(state='disabled')
        self.btn_detener_captura.config(state='normal')
        self.btn_registrar.config(state='disabled')
        
        self.label_estado_registro.config(text="Capturando... Mire a la cámara")
        self.actualizar_vista_registro()
    
    def detener_captura_registro(self):
        """Detiene el proceso de captura para registro"""
        self.capturando = False
        self.btn_iniciar_captura.config(state='normal')
        self.btn_detener_captura.config(state='disabled')
        
        if self.reconocedor.capturas_realizadas >= self.reconocedor.capturas_por_registro:
            self.btn_registrar.config(state='normal')
            self.label_estado_registro.config(text="Captura completada. Puede registrar.")
        else:
            self.label_estado_registro.config(text="Captura detenida. Capturas insuficientes.")
    
    def actualizar_vista_registro(self):
        """Actualiza la vista de cámara en la pestaña de registro"""
        if not self.capturando or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if ret:
            # Procesar frame para captura
            captura_exitosa, ubicacion = self.reconocedor.capturar_para_registro(frame)
            
            # Dibujar rectángulo alrededor del rostro
            if ubicacion:
                top, right, bottom, left = ubicacion
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"Captura {self.reconocedor.capturas_realizadas}/3", 
                           (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convertir para mostrar en Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            
            self.label_camara.img_tk = img_tk
            self.label_camara.config(image=img_tk)
            
            # Actualizar progreso
            self.progress_registro['value'] = self.reconocedor.capturas_realizadas
            self.label_progreso.config(
                text=f"Capturas: {self.reconocedor.capturas_realizadas}/{self.reconocedor.capturas_por_registro}"
            )
            
            # Continuar captura si es necesario
            if self.reconocedor.capturas_realizadas < self.reconocedor.capturas_por_registro:
                self.root.after(100, self.actualizar_vista_registro)  # Más lento para mejor captura
            else:
                self.detener_captura_registro()
    
    def registrar_persona(self):
        """Registra una nueva persona en el sistema"""
        nombre = self.entry_nombre.get()
        apellido = self.entry_apellido.get()
        email = self.entry_email.get()
        
        # Obtener embedding promedio
        embedding = self.reconocedor.finalizar_registro()
        if embedding is None:
            messagebox.showerror("Error", "No hay suficientes capturas para registrar")
            return
        
        # Registrar en base de datos
        exito, mensaje = self.db.registrar_persona(nombre, apellido, email, embedding)
        
        if exito:
            messagebox.showinfo("Éxito", mensaje)
            self.limpiar_formulario_registro()
            self.actualizar_lista_personas()
        else:
            messagebox.showerror("Error", mensaje)
    
    def limpiar_formulario_registro(self):
        """Limpia el formulario de registro"""
        self.entry_nombre.delete(0, tk.END)
        self.entry_apellido.delete(0, tk.END)
        self.entry_email.delete(0, tk.END)
        self.progress_registro['value'] = 0
        self.label_progreso.config(text="Capturas: 0/3")
        self.label_estado_registro.config(text="Listo para capturar")
    
    def iniciar_deteccion(self):
        """Inicia el proceso de detección en tiempo real"""
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Cámara no disponible")
            return
        
        self.detectando = True
        self.btn_iniciar_deteccion.config(state='disabled')
        self.btn_detener_deteccion.config(state='normal')
        
        # Reiniciar contador de frames
        self.frame_count = 0
        
        self.actualizar_vista_deteccion()
    
    def detener_deteccion(self):
        """Detiene el proceso de detección"""
        self.detectando = False
        self.btn_iniciar_deteccion.config(state='normal')
        self.btn_detener_deteccion.config(state='disabled')
        
        # Limpiar información de detección
        self.label_info_persona.config(text="Persona: No detectada")
        self.label_info_emocion.config(text="Emoción: -")
        self.label_info_confianza.config(text="Confianza: -")
    
    def actualizar_vista_deteccion(self):
        """Actualiza la vista de detección en tiempo real - OPTIMIZADO CON FER"""
        if not self.detectando or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            
            # Procesar cada 2 frames para mejor rendimiento (FER puede ser más lento)
            procesar_este_frame = (self.frame_count % 2 == 0)
            
            frame_procesado = frame.copy()
            
            if procesar_este_frame:
                # Realizar reconocimiento facial
                persona, confianza, ubicacion = self.reconocedor.reconocer_persona(frame)
                
                if persona and ubicacion:
                    # Dibujar rectángulo alrededor del rostro
                    top, right, bottom, left = ubicacion
                    cv2.rectangle(frame_procesado, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Predecir emoción usando FER
                    emocion, confianza_emocion = self.analizador.predecir_emocion(frame, ubicacion)
                    
                    # Aplicar suavizado al historial de emociones
                    emocion_suavizada, confianza_suavizada = self.suavizar_emocion(emocion, confianza_emocion)
                    
                    # Registrar detección en base de datos (solo cada 15 frames para no saturar)
                    if self.frame_count % 15 == 0:
                        self.db.registrar_deteccion(persona.id, emocion_suavizada, confianza_suavizada)
                    
                    # Actualizar información en interfaz
                    self.label_info_persona.config(text=f"Persona: {persona.nombre} {persona.apellido}")
                    self.label_info_emocion.config(text=f"Emoción: {emocion_suavizada}")
                    self.label_info_confianza.config(
                        text=f"Reconocimiento: {confianza:.1f}% - Emoción: {confianza_suavizada:.1%}"
                    )
                    
                    # Añadir texto al frame
                    cv2.putText(frame_procesado, f"{persona.nombre}", 
                               (left, top-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame_procesado, f"{emocion_suavizada} ({confianza_suavizada:.1%})", 
                               (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    self.label_info_persona.config(text="Persona: No detectada")
                    self.label_info_emocion.config(text="Emoción: -")
                    self.label_info_confianza.config(text="Confianza: -")
                    
                    if ubicacion:
                        top, right, bottom, left = ubicacion
                        cv2.rectangle(frame_procesado, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame_procesado, "Desconocido", 
                                   (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Convertir para mostrar en Tkinter
            frame_rgb = cv2.cvtColor(frame_procesado, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            
            self.label_camara_deteccion.img_tk = img_tk
            self.label_camara_deteccion.config(image=img_tk)
            
            # Continuar detección con delay ajustado para FER
            self.root.after(30, self.actualizar_vista_deteccion)
    
    def suavizar_emocion(self, emocion_actual, confianza_actual):
        """Suaviza las emociones usando un historial para evitar cambios bruscos"""
        # Agregar al historial
        self.historial_emociones.append((emocion_actual, confianza_actual))
        
        # Mantener solo los últimos N elementos
        if len(self.historial_emociones) > self.max_historial:
            self.historial_emociones.pop(0)
        
        if not self.historial_emociones:
            return emocion_actual, confianza_actual
        
        # Contar frecuencias de emociones en el historial
        conteo_emociones = {}
        suma_confianzas = {}
        
        for emocion, confianza in self.historial_emociones:
            if emocion not in conteo_emociones:
                conteo_emociones[emocion] = 0
                suma_confianzas[emocion] = 0.0
            
            conteo_emociones[emocion] += 1
            suma_confianzas[emocion] += confianza
        
        # Encontrar la emoción más frecuente
        emocion_dominante = max(conteo_emociones, key=conteo_emociones.get)
        
        # Calcular confianza promedio para esa emoción
        confianza_promedio = suma_confianzas[emocion_dominante] / conteo_emociones[emocion_dominante]
        
        return emocion_dominante, confianza_promedio
    
    def actualizar_cache_sistema(self):
        """Actualiza la cache del sistema para reconocimiento más rápido"""
        self.reconocedor.actualizar_cache()
        messagebox.showinfo("Éxito", "Cache del sistema actualizado")
    
    def actualizar_lista_personas(self):
        """Actualiza la lista de personas en el combobox de reportes"""
        personas = self.db.obtener_todas_personas()
        nombres_personas = [f"{p.nombre} {p.apellido} (ID: {p.id})" for p in personas]
        self.combo_personas['values'] = nombres_personas
        
        if nombres_personas:
            self.combo_personas.set(nombres_personas[0])
        else:
            self.combo_personas.set('')
    
    def generar_reporte_persona(self):
        """Genera un reporte PDF para la persona seleccionada"""
        seleccion = self.combo_personas.get()
        if not seleccion:
            messagebox.showwarning("Advertencia", "Seleccione una persona")
            return
        
        # Extraer ID de la persona
        try:
            persona_id = int(seleccion.split("ID: ")[1].rstrip(')'))
        except:
            messagebox.showerror("Error", "Formato de selección inválido")
            return
        
        # Seleccionar ubicación para guardar
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Guardar reporte de persona"
        )
        
        if file_path:
            exito, mensaje = self.generador_reportes.generar_reporte_persona(persona_id, file_path)
            if exito:
                messagebox.showinfo("Éxito", mensaje)
                self.mostrar_resumen_reporte(persona_id)
            else:
                messagebox.showerror("Error", mensaje)
    
    def generar_reporte_general(self):
        """Genera un reporte PDF general del sistema"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Guardar reporte general"
        )
        
        if file_path:
            exito, mensaje = self.generador_reportes.generar_reporte_general(file_path)
            if exito:
                messagebox.showinfo("Éxito", mensaje)
                self.mostrar_resumen_general()
            else:
                messagebox.showerror("Error", mensaje)
    
    def mostrar_resumen_reporte(self, persona_id):
        """Muestra un resumen del reporte en el área de texto"""
        persona = self.db.obtener_persona_por_id(persona_id)
        if not persona:
            return
        
        estadisticas = self.db.obtener_estadisticas_emociones(persona_id)
        total = sum(estadisticas.values())
        
        resumen = f"RESUMEN - {persona.nombre} {persona.apellido}\n"
        resumen += "=" * 50 + "\n\n"
        resumen += f"Total de detecciones: {total}\n\n"
        resumen += "Distribución de emociones:\n"
        
        for emocion, count in estadisticas.items():
            porcentaje = (count / total * 100) if total > 0 else 0
            resumen += f"  {emocion}: {count} ({porcentaje:.1f}%)\n"
        
        self.texto_reportes.delete(1.0, tk.END)
        self.texto_reportes.insert(1.0, resumen)
    
    def mostrar_resumen_general(self):
        """Muestra un resumen general en el área de texto"""
        personas = self.db.obtener_todas_personas()
        estadisticas = self.db.obtener_estadisticas_emociones()
        total_detecciones = sum(estadisticas.values())
        
        resumen = "REPORTE GENERAL DEL SISTEMA\n"
        resumen += "=" * 50 + "\n\n"
        resumen += f"Personas registradas: {len(personas)}\n"
        resumen += f"Total de detecciones: {total_detecciones}\n\n"
        resumen += "Distribución general de emociones:\n"
        
        for emocion, count in estadisticas.items():
            porcentaje = (count / total_detecciones * 100) if total_detecciones > 0 else 0
            resumen += f"  {emocion}: {count} ({porcentaje:.1f}%)\n"
        
        self.texto_reportes.delete(1.0, tk.END)
        self.texto_reportes.insert(1.0, resumen)
    
    def __del__(self):
        """Liberar recursos al cerrar la aplicación"""
        self.procesamiento_activo = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()