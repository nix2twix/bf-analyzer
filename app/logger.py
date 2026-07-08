# logger.py
from datetime import datetime
import os
import sys

class AppLogger:
    """Класс для логирования действий приложения"""
    
    def __init__(self):
        self.log_file = "app_logs.txt"
        # Создаем файл логов если его нет, с UTF-8 кодировкой
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Application started at {datetime.now()} ===\n")
    
    def _safe_encode(self, text):
        """Безопасное кодирование текста для Windows консоли"""
        try:
            # Пытаемся закодировать в cp1251, заменяя неподдерживаемые символы
            return text.encode('cp1251', errors='replace').decode('cp1251')
        except:
            return text.encode('ascii', errors='replace').decode('ascii')
    
    def log(self, action, details=None, level="INFO"):
        """Универсальный метод логирования"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = f"[{timestamp}] [{level}] {action}"
        if details:
            log_entry += f" | {details}"
        
        # Выводим в консоль с безопасным кодированием для Windows
        try:
            print(log_entry)
        except UnicodeEncodeError:
            safe_entry = self._safe_encode(log_entry)
            print(safe_entry)
        
        # Сохраняем в файл с UTF-8 (всегда работает)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            # Если не можем записать в файл, хотя бы выведем ошибку
            print(f"[ERROR] Failed to write log: {e}")
    
    def log_image_upload(self, filename, width, height, file_size_mb):
        """Логирование загрузки изображения"""
        self.log(
            "IMAGE_UPLOAD",
            f"name='{filename}', size={width}x{height}, file_size={file_size_mb:.2f}MB",
            "INFO"
        )
    
    def log_segmentation_start(self, filename, model, scale_info):
        """Логирование начала сегментации"""
        self.log(
            "SEGMENTATION_START",
            f"image='{filename}', model='{model}', scale={scale_info}",
            "INFO"
        )
    
    def log_segmentation_complete(self, filename, objects_count, elapsed_time, model):
        """Логирование завершения сегментации"""
        self.log(
            "SEGMENTATION_COMPLETE",
            f"image='{filename}', objects={objects_count}, time={elapsed_time:.2f}s, model='{model}'",
            "SUCCESS"
        )
    
    def log_segmentation_error(self, filename, error, model):
        """Логирование ошибки сегментации"""
        self.log(
            "SEGMENTATION_ERROR",
            f"image='{filename}', error='{str(error)}', model='{model}'",
            "ERROR"
        )
    
    def log_scale_detection(self, filename, scale_value, detected_text):
        """Логирование определения масштаба"""
        self.log(
            "SCALE_DETECTION",
            f"image='{filename}', scale={scale_value}, detected_text='{detected_text}'",
            "INFO"
        )
    
    def log_export(self, filename, export_type, file_size):
        """Логирование экспорта"""
        self.log(
            "EXPORT",
            f"image='{filename}', type='{export_type}', size={file_size:.2f}KB",
            "INFO"
        )
    
    def log_error(self, action, error, filename=None):
        """Логирование ошибок"""
        details = f"action='{action}', error='{str(error)}'"
        if filename:
            details += f", image='{filename}'"
        self.log("ERROR", details, "ERROR")

# Глобальный экземпляр логгера
logger = AppLogger()