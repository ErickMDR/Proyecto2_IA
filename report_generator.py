import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from database import DatabaseManager, Persona
from datetime import datetime, timedelta
import numpy as np

class GeneradorReportes:
    def __init__(self, db_manager):
        self.db = db_manager
    
    def generar_reporte_persona(self, persona_id, output_path):
        """Genera un reporte PDF para una persona específica"""
        persona = self.db.obtener_persona_por_id(persona_id)
        if not persona:
            return False, "Persona no encontrada"
        
        try:
            with PdfPages(output_path) as pdf:
                # Página 1: Resumen y estadísticas
                self._generar_pagina_resumen(pdf, persona)
                
                # Página 2: Gráfico de emociones
                self._generar_pagina_grafico(pdf, persona)
                
                # Página 3: Historial reciente
                self._generar_pagina_historial(pdf, persona)
            
            return True, f"Reporte generado: {output_path}"
        except Exception as e:
            return False, f"Error al generar reporte: {str(e)}"
    
    def generar_reporte_general(self, output_path, dias=30):
        """Genera un reporte PDF general del sistema"""
        try:
            with PdfPages(output_path) as pdf:
                # Página 1: Estadísticas generales
                self._generar_pagina_estadisticas_generales(pdf, dias)
                
                # Página 2: Distribución de emociones
                self._generar_pagina_distribucion_emociones(pdf, dias)
            
            return True, f"Reporte general generado: {output_path}"
        except Exception as e:
            return False, f"Error al generar reporte general: {str(e)}"
    
    def _generar_pagina_resumen(self, pdf, persona):
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(f'Reporte de Emociones - {persona.nombre} {persona.apellido}', fontsize=16)
        
        # Estadísticas
        estadisticas = self.db.obtener_estadisticas_emociones(persona.id)
        total_detecciones = sum(estadisticas.values())
        
        ax.axis('off')  # Ocultar ejes
        
        info_text = f"""
        Información Personal:
        Nombre: {persona.nombre} {persona.apellido}
        Email: {persona.email}
        Fecha de Registro: {persona.fecha_registro.strftime('%Y-%m-%d')}
        
        Estadísticas de Emociones:
        Total de Detecciones: {total_detecciones}
        """
        
        for emocion, count in estadisticas.items():
            porcentaje = (count / total_detecciones * 100) if total_detecciones > 0 else 0
            info_text += f"{emocion}: {count} ({porcentaje:.1f}%)\n"
        
        ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', linespacing=1.5)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _generar_pagina_grafico(self, pdf, persona):
        estadisticas = self.db.obtener_estadisticas_emociones(persona.id)
        
        if not estadisticas:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No hay datos de emociones para esta persona", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Distribución de Emociones - {persona.nombre} {persona.apellido}')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        emociones = list(estadisticas.keys())
        conteos = list(estadisticas.values())
        
        colores = ['red', 'orange', 'purple', 'green', 'blue', 'yellow', 'gray']
        
        bars = ax.bar(emociones, conteos, color=colores[:len(emociones)])
        ax.set_title(f'Distribución de Emociones - {persona.nombre} {persona.apellido}')
        ax.set_ylabel('Número de Detecciones')
        ax.set_xlabel('Emociones')
        
        # Añadir valores en las barras
        for bar, count in zip(bars, conteos):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _generar_pagina_historial(self, pdf, persona):
        historial = self.db.obtener_historial_emociones(persona.id, dias=30)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.set_title(f'Historial Reciente - {persona.nombre} {persona.apellido}')
        
        if not historial:
            ax.text(0.5, 0.5, "No hay datos de detección recientes", 
                   ha='center', va='center', transform=ax.transAxes)
        else:
            # Mostrar últimas 10 detecciones
            historial_reciente = historial[-10:] if len(historial) > 10 else historial
            
            texto_historial = "Últimas Detecciones:\n\n"
            for deteccion in historial_reciente:
                fecha = deteccion.fecha_deteccion.strftime('%Y-%m-%d %H:%M')
                texto_historial += f"{fecha}: {deteccion.emocion} ({deteccion.confianza:.1%})\n"
            
            ax.text(0.1, 0.9, texto_historial, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', linespacing=1.5)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _generar_pagina_estadisticas_generales(self, pdf, dias):
        personas = self.db.obtener_todas_personas()
        total_detecciones = len(self.db.obtener_historial_emociones(dias=dias))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        ax.set_title('Reporte General del Sistema', fontsize=16)
        
        info_general = f"""
        Estadísticas Generales:
        Personas Registradas: {len(personas)}
        Total de Detecciones (Últimos {dias} días): {total_detecciones}
        Fecha del Reporte: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        
        Personas Registradas:
        """
        
        for persona in personas:
            detecciones_persona = len(self.db.obtener_historial_emociones(persona.id, dias))
            info_general += f"- {persona.nombre} {persona.apellido} ({detecciones_persona} detecciones)\n"
        
        ax.text(0.1, 0.9, info_general, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', linespacing=1.5)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _generar_pagina_distribucion_emociones(self, pdf, dias):
        estadisticas = self.db.obtener_estadisticas_emociones()
        
        if not estadisticas:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "No hay datos de emociones en el sistema", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Distribución General de Emociones (Últimos {dias} días)')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Distribución General de Emociones (Últimos {dias} días)')
        
        emociones = list(estadisticas.keys())
        conteos = list(estadisticas.values())
        total = sum(conteos)
        
        colores = ['red', 'orange', 'purple', 'green', 'blue', 'yellow', 'gray']
        
        # Gráfico de barras
        bars = ax1.bar(emociones, conteos, color=colores[:len(emociones)])
        ax1.set_title('Detecciones por Emoción')
        ax1.set_ylabel('Número de Detecciones')
        ax1.tick_params(axis='x', rotation=45)
        
        # Gráfico de pie
        if total > 0:
            ax2.pie(conteos, labels=emociones, autopct='%1.1f%%', 
                   colors=colores[:len(emociones)])
            ax2.set_title('Distribución Porcentual')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()