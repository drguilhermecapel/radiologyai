"""
Sistema avançado de download de modelos para MedAI Radiologia
Implementa download com GUI, retry automático, e fallback inteligente
"""

import os
import sys
import json
import time
import hashlib
import requests
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from urllib.parse import urlparse
import logging

logger = logging.getLogger('MedAI.ModelDownloader')

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    logger.warning("⚠️ GUI não disponível, usando modo console")

class ModelDownloader:
    """
    Sistema avançado de download de modelos com interface gráfica
    Suporta retry automático, resume de downloads e fallback offline
    """
    
    def __init__(self, base_path: Optional[str] = None, use_gui: bool = True):
        """
        Inicializa o sistema de download
        
        Args:
            base_path: Caminho base para modelos
            use_gui: Se deve usar interface gráfica
        """
        if base_path is None:
            current_dir = Path(__file__).parent
            self.base_path = current_dir.parent / "models"
        else:
            self.base_path = Path(base_path)
        
        self.registry_path = self.base_path / "model_registry.json"
        self.download_cache_path = self.base_path / ".download_cache"
        
        self.chunk_size = 8192
        self.timeout = 300
        self.max_retries = 3
        self.retry_delay = 2
        
        self.current_downloads = {}
        self.download_stats = {
            'total_downloaded': 0,
            'total_size': 0,
            'failed_downloads': 0,
            'successful_downloads': 0
        }
        
        self.use_gui = use_gui and GUI_AVAILABLE
        self.gui_window = None
        self.progress_widgets = {}
        
        self.progress_callback = None
        self.completion_callback = None
        
        self._load_registry()
        self._ensure_directories()
    
    def _load_registry(self):
        """Carrega registro de modelos"""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    self.registry = json.load(f)
                logger.info(f"✅ Registro carregado: {len(self.registry.get('models', {}))} modelos")
            else:
                logger.warning("⚠️ Registro de modelos não encontrado")
                self.registry = {'models': {}}
        except Exception as e:
            logger.error(f"❌ Erro ao carregar registro: {e}")
            self.registry = {'models': {}}
    
    def _ensure_directories(self):
        """Garante que diretórios necessários existem"""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.download_cache_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model_info in self.registry.get('models', {}).items():
            model_dir = self.base_path / Path(model_info['file_path']).parent
            model_dir.mkdir(parents=True, exist_ok=True)
    
    def download_all_models(self, models: Optional[List[str]] = None, 
                          show_gui: bool = True) -> Dict[str, bool]:
        """
        Baixa todos os modelos ou lista específica
        
        Args:
            models: Lista de modelos para baixar (None = todos)
            show_gui: Se deve mostrar interface gráfica
            
        Returns:
            Dict com resultado de cada download
        """
        if models is None:
            models = list(self.registry.get('models', {}).keys())
        
        if not models:
            logger.warning("⚠️ Nenhum modelo para baixar")
            return {}
        
        logger.info(f"📥 Iniciando download de {len(models)} modelos")
        
        if show_gui and self.use_gui:
            self._setup_download_gui(models)
        
        results = {}
        
        if self.use_gui and show_gui:
            results = self._download_with_gui(models)
        else:
            results = self._download_console(models)
        
        self._update_download_stats(results)
        
        return results
    
    def _setup_download_gui(self, models: List[str]):
        """Configura interface gráfica de download"""
        if not self.use_gui:
            return
        
        self.gui_window = tk.Tk()
        self.gui_window.title("MedAI - Download de Modelos Pré-treinados")
        self.gui_window.geometry("600x400")
        self.gui_window.resizable(True, True)
        
        style = ttk.Style()
        style.theme_use('clam')
        
        main_frame = ttk.Frame(self.gui_window, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.gui_window.columnconfigure(0, weight=1)
        self.gui_window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        title_label = ttk.Label(main_frame, text="Download de Modelos de IA Médica", 
                               font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        general_frame = ttk.LabelFrame(main_frame, text="Progresso Geral", padding="10")
        general_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        general_frame.columnconfigure(1, weight=1)
        
        ttk.Label(general_frame, text="Total:").grid(row=0, column=0, sticky=tk.W)
        self.general_progress = ttk.Progressbar(general_frame, length=400, mode='determinate')
        self.general_progress.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        
        self.general_label = ttk.Label(general_frame, text="0%")
        self.general_label.grid(row=0, column=2, padx=(10, 0))
        
        models_frame = ttk.LabelFrame(main_frame, text="Modelos Individuais", padding="10")
        models_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        models_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        canvas = tk.Canvas(models_frame, height=200)
        scrollbar = ttk.Scrollbar(models_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        models_frame.rowconfigure(0, weight=1)
        models_frame.columnconfigure(0, weight=1)
        
        self.progress_widgets = {}
        for i, model_name in enumerate(models):
            model_info = self.registry['models'].get(model_name, {})
            
            model_frame = ttk.Frame(scrollable_frame)
            model_frame.grid(row=i, column=0, sticky="ew", pady=2)
            model_frame.columnconfigure(1, weight=1)
            
            name_label = ttk.Label(model_frame, text=model_info.get('name', model_name))
            name_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
            
            progress_bar = ttk.Progressbar(model_frame, length=300, mode='determinate')
            progress_bar.grid(row=0, column=1, sticky="ew")
            
            status_label = ttk.Label(model_frame, text="Aguardando...")
            status_label.grid(row=0, column=2, padx=(10, 0))
            
            self.progress_widgets[model_name] = {
                'progress_bar': progress_bar,
                'status_label': status_label,
                'name_label': name_label
            }
        
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        
        self.cancel_button = ttk.Button(controls_frame, text="Cancelar", 
                                       command=self._cancel_downloads)
        self.cancel_button.grid(row=0, column=0, padx=(0, 10))
        
        self.close_button = ttk.Button(controls_frame, text="Fechar", 
                                      command=self._close_gui, state='disabled')
        self.close_button.grid(row=0, column=1)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Preparando downloads...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        
        self.gui_window.update_idletasks()
        x = (self.gui_window.winfo_screenwidth() // 2) - (self.gui_window.winfo_width() // 2)
        y = (self.gui_window.winfo_screenheight() // 2) - (self.gui_window.winfo_height() // 2)
        self.gui_window.geometry(f"+{x}+{y}")
        
        self.gui_window.protocol("WM_DELETE_WINDOW", self._on_window_close)
    
    def _download_with_gui(self, models: List[str]) -> Dict[str, bool]:
        """Executa downloads com interface gráfica"""
        results = {}
        self.download_cancelled = False
        
        def download_thread():
            try:
                total_models = len(models)
                
                for i, model_name in enumerate(models):
                    if self.download_cancelled:
                        break
                    
                    general_progress = (i / total_models) * 100
                    if self.gui_window:
                        self.gui_window.after(0, self._update_general_progress, 
                                            general_progress, f"Baixando {model_name}...")
                    
                    success = self._download_single_model(model_name, gui_mode=True)
                    results[model_name] = success
                    
                    status = "✅ Concluído" if success else "❌ Falhou"
                    if self.gui_window:
                        self.gui_window.after(0, self._update_model_status, model_name, 100, status)
                
                if not self.download_cancelled:
                    if self.gui_window:
                        self.gui_window.after(0, self._download_completed, results)
                else:
                    if self.gui_window:
                        self.gui_window.after(0, self._download_cancelled_gui)
                    
            except Exception as e:
                logger.error(f"❌ Erro no thread de download: {e}")
                if self.gui_window:
                    self.gui_window.after(0, self._download_error, str(e))
        
        download_thread_obj = threading.Thread(target=download_thread, daemon=True)
        download_thread_obj.start()
        
        if self.gui_window:
            self.gui_window.mainloop()
        
        return results
    
    def _download_console(self, models: List[str]) -> Dict[str, bool]:
        """Executa downloads em modo console"""
        results = {}
        total_models = len(models)
        
        print(f"\n📥 Baixando {total_models} modelos...")
        print("=" * 60)
        
        for i, model_name in enumerate(models, 1):
            print(f"\n[{i}/{total_models}] {model_name}")
            print("-" * 40)
            
            success = self._download_single_model(model_name, gui_mode=False)
            results[model_name] = success
            
            status = "✅ SUCESSO" if success else "❌ FALHOU"
            print(f"Status: {status}")
        
        print("\n" + "=" * 60)
        successful = sum(1 for success in results.values() if success)
        print(f"📊 Resultado: {successful}/{total_models} modelos baixados com sucesso")
        
        return results
    
    def _download_single_model(self, model_name: str, gui_mode: bool = False) -> bool:
        """
        Baixa um modelo específico
        
        Args:
            model_name: Nome do modelo
            gui_mode: Se está em modo GUI
            
        Returns:
            True se download foi bem-sucedido
        """
        if model_name not in self.registry.get('models', {}):
            logger.error(f"❌ Modelo {model_name} não encontrado no registro")
            return False
        
        model_info = self.registry['models'][model_name]
        model_path = self.base_path / model_info['file_path']
        
        if self._verify_model_integrity(model_name, model_path, model_info):
            if gui_mode and self.gui_window:
                self.gui_window.after(0, self._update_model_status, 
                                    model_name, 100, "✅ Já existe")
            else:
                print("✅ Modelo já existe e está íntegro")
            return True
        
        urls = [model_info['download_url']] + model_info.get('backup_urls', [])
        
        for attempt in range(self.max_retries):
            for url_index, url in enumerate(urls):
                try:
                    if gui_mode and self.gui_window:
                        status = f"Tentativa {attempt + 1}/{self.max_retries} (URL {url_index + 1})"
                        self.gui_window.after(0, self._update_model_status, 
                                            model_name, 0, status)
                    else:
                        print(f"📥 Tentativa {attempt + 1}/{self.max_retries} - URL {url_index + 1}")
                    
                    success = self._download_file(url, model_path, model_info, 
                                                model_name, gui_mode)
                    
                    if success:
                        if self._verify_model_integrity(model_name, model_path, model_info):
                            return True
                        else:
                            logger.warning(f"⚠️ Modelo {model_name} falhou na verificação")
                            model_path.unlink(missing_ok=True)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Erro no download de {url}: {e}")
                    continue
            
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)  # Backoff exponencial
                if gui_mode and self.gui_window:
                    self.gui_window.after(0, self._update_model_status, 
                                        model_name, 0, f"Aguardando {delay}s...")
                else:
                    print(f"⏳ Aguardando {delay}s...")
                time.sleep(delay)
        
        logger.error(f"❌ Falha no download do modelo {model_name}")
        return False
    
    def _download_file(self, url: str, file_path: Path, model_info: Dict, 
                      model_name: str, gui_mode: bool) -> bool:
        """
        Baixa arquivo específico com progresso
        
        Args:
            url: URL para download
            file_path: Caminho local
            model_info: Informações do modelo
            model_name: Nome do modelo
            gui_mode: Se está em modo GUI
            
        Returns:
            True se download foi bem-sucedido
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            response = requests.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            expected_size = model_info.get('file_size', 0)
            
            if expected_size > 0 and total_size > 0:
                size_diff = abs(total_size - expected_size)
                if size_diff > 1024:  # Tolerância de 1KB
                    logger.warning(f"⚠️ Tamanho não confere: esperado {expected_size}, obtido {total_size}")
            
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            
                            if gui_mode and self.gui_window:
                                status = f"{downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB"
                                self.gui_window.after(0, self._update_model_progress, 
                                                    model_name, progress, status)
                            else:
                                self._print_progress(downloaded, total_size)
            
            if not gui_mode:
                print()  # Nova linha após progresso
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro no download: {e}")
            file_path.unlink(missing_ok=True)
            return False
    
    def _verify_model_integrity(self, model_name: str, file_path: Path, 
                               model_info: Dict) -> bool:
        """Verifica integridade do modelo"""
        if not file_path.exists():
            return False
        
        try:
            file_size = file_path.stat().st_size
            expected_size = model_info.get('file_size', 0)
            
            if expected_size > 0:
                size_diff = abs(file_size - expected_size)
                if size_diff > 1024:  # Tolerância de 1KB
                    logger.warning(f"⚠️ Tamanho incorreto para {model_name}")
                    return False
            
            expected_hash = model_info.get('sha256_hash')
            if expected_hash:
                actual_hash = self._calculate_file_hash(file_path)
                if actual_hash != expected_hash:
                    logger.error(f"❌ Hash incorreto para {model_name}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na verificação de {model_name}: {e}")
            return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calcula hash SHA256 do arquivo"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _print_progress(self, downloaded: int, total: int):
        """Imprime progresso no console"""
        if total > 0:
            progress = (downloaded / total) * 100
            bar_length = 40
            filled_length = int(bar_length * downloaded / total)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            
            print(f"\r📥 |{bar}| {progress:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", 
                  end='', flush=True)
    
    def _update_general_progress(self, progress: float, status: str):
        """Atualiza progresso geral na GUI"""
        if hasattr(self, 'general_progress'):
            self.general_progress['value'] = progress
            self.general_label.config(text=f"{progress:.1f}%")
            self.status_var.set(status)
    
    def _update_model_progress(self, model_name: str, progress: float, status: str):
        """Atualiza progresso de modelo específico"""
        if model_name in self.progress_widgets:
            widgets = self.progress_widgets[model_name]
            widgets['progress_bar']['value'] = progress
            widgets['status_label'].config(text=status)
    
    def _update_model_status(self, model_name: str, progress: float, status: str):
        """Atualiza status de modelo específico"""
        if model_name in self.progress_widgets:
            widgets = self.progress_widgets[model_name]
            widgets['progress_bar']['value'] = progress
            widgets['status_label'].config(text=status)
    
    def _download_completed(self, results: Dict[str, bool]):
        """Callback quando download é concluído"""
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        self._update_general_progress(100, f"Concluído: {successful}/{total} modelos")
        
        self.cancel_button.config(state='disabled')
        self.close_button.config(state='normal')
        
        if successful == total:
            messagebox.showinfo("Sucesso", 
                              f"Todos os {total} modelos foram baixados com sucesso!")
        else:
            failed = total - successful
            messagebox.showwarning("Parcialmente Concluído", 
                                 f"{successful} modelos baixados com sucesso.\n"
                                 f"{failed} modelos falharam.")
    
    def _download_cancelled_gui(self):
        """Callback quando download é cancelado"""
        self.status_var.set("Download cancelado pelo usuário")
        self.cancel_button.config(state='disabled')
        self.close_button.config(state='normal')
        
        messagebox.showinfo("Cancelado", "Download cancelado pelo usuário.")
    
    def _download_error(self, error_msg: str):
        """Callback quando ocorre erro no download"""
        self.status_var.set(f"Erro: {error_msg}")
        self.cancel_button.config(state='disabled')
        self.close_button.config(state='normal')
        
        messagebox.showerror("Erro", f"Erro durante o download:\n{error_msg}")
    
    def _cancel_downloads(self):
        """Cancela downloads em andamento"""
        self.download_cancelled = True
        self.cancel_button.config(state='disabled')
        self.status_var.set("Cancelando downloads...")
    
    def _close_gui(self):
        """Fecha interface gráfica"""
        if self.gui_window:
            self.gui_window.destroy()
    
    def _on_window_close(self):
        """Callback quando janela é fechada"""
        if hasattr(self, 'download_cancelled'):
            self.download_cancelled = True
        self._close_gui()
    
    def _update_download_stats(self, results: Dict[str, bool]):
        """Atualiza estatísticas de download"""
        for model_name, success in results.items():
            if success:
                self.download_stats['successful_downloads'] += 1
                model_info = self.registry['models'].get(model_name, {})
                size = model_info.get('file_size', 0)
                self.download_stats['total_downloaded'] += size
                self.download_stats['total_size'] += size
            else:
                self.download_stats['failed_downloads'] += 1
    
    def get_download_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de download"""
        return self.download_stats.copy()
    
    def check_internet_connection(self) -> bool:
        """Verifica se há conexão com internet"""
        try:
            response = requests.get("https://www.google.com", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def estimate_download_time(self, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Estima tempo de download baseado no tamanho dos modelos
        
        Args:
            models: Lista de modelos (None = todos)
            
        Returns:
            Dict com estimativas de tempo
        """
        if models is None:
            models = list(self.registry.get('models', {}).keys())
        
        total_size = 0
        model_sizes = {}
        
        for model_name in models:
            model_info = self.registry['models'].get(model_name, {})
            size = model_info.get('file_size', 0)
            total_size += size
            model_sizes[model_name] = size
        
        speeds = {
            'dial_up': 56 * 1024 / 8,      # 56k modem
            'broadband': 1 * 1024 * 1024,   # 1 Mbps
            'fast': 10 * 1024 * 1024,       # 10 Mbps
            'very_fast': 100 * 1024 * 1024  # 100 Mbps
        }
        
        estimates = {}
        for speed_name, speed_bps in speeds.items():
            time_seconds = total_size / speed_bps
            estimates[speed_name] = {
                'seconds': time_seconds,
                'minutes': time_seconds / 60,
                'hours': time_seconds / 3600,
                'formatted': self._format_time(time_seconds)
            }
        
        return {
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'total_size_gb': total_size / (1024 * 1024 * 1024),
            'model_count': len(models),
            'model_sizes': model_sizes,
            'time_estimates': estimates
        }
    
    def _format_time(self, seconds: float) -> str:
        """Formata tempo em string legível"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}min"
        else:
            hours = seconds / 3600
            if hours < 24:
                return f"{hours:.1f}h"
            else:
                days = hours / 24
                return f"{days:.1f}d"


class OfflineFallbackManager:
    """
    Gerenciador de fallback para modo offline
    Fornece modelos básicos quando download não é possível
    """
    
    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            current_dir = Path(__file__).parent
            self.base_path = current_dir.parent / "models" / "fallback"
        else:
            self.base_path = Path(base_path)
        
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def create_basic_models(self) -> Dict[str, str]:
        """
        Cria modelos básicos para fallback offline
        
        Returns:
            Dict com caminhos dos modelos criados
        """
        models_created = {}
        
        try:
            import tensorflow as tf
            
            basic_model = self._create_basic_chest_model()
            basic_path = self.base_path / "basic_chest_model.h5"
            basic_model.save(str(basic_path))
            models_created['basic_chest'] = str(basic_path)
            
            logger.info("✅ Modelos básicos de fallback criados")
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar modelos de fallback: {e}")
        
        return models_created
    
    def _create_basic_chest_model(self):
        """Cria modelo básico para análise de chest X-ray"""
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        model = models.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(5, activation='softmax')  # 5 classes básicas
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def is_offline_mode_needed(self) -> bool:
        """Verifica se modo offline é necessário"""
        downloader = ModelDownloader()
        return not downloader.check_internet_connection()
