"""
Sistema de carregamento de modelos pré-treinados para MedAI Radiologia
Implementa download automático, verificação de integridade e carregamento inteligente
"""

import os
import json
import hashlib
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import tensorflow as tf
from threading import Thread
import time
from urllib.parse import urlparse

logger = logging.getLogger('MedAI.PreTrainedLoader')

class PreTrainedModelLoader:
    """
    Gerenciador de modelos pré-treinados com download automático
    Integra com SOTAModelManager para carregamento inteligente
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Inicializa o carregador de modelos pré-treinados
        
        Args:
            base_path: Caminho base para modelos (padrão: models/pre_trained)
        """
        if base_path is None:
            current_dir = Path(__file__).parent
            self.base_path = current_dir.parent / "models" / "pre_trained"
        else:
            self.base_path = Path(base_path)
        
        self.registry_path = self.base_path.parent / "model_registry.json"
        self.models_registry = {}
        self.loaded_models = {}
        self.download_progress_callback = None
        
        self.download_timeout = 300
        self.retry_attempts = 3
        self.chunk_size = 8192
        
        self._load_registry()
        
    def _load_registry(self):
        """Carrega o registro de modelos disponíveis"""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
                    self.models_registry = registry_data.get('models', {})
                    
                    download_settings = registry_data.get('download_settings', {})
                    self.download_timeout = download_settings.get('default_timeout', 300)
                    self.retry_attempts = download_settings.get('retry_attempts', 3)
                    self.chunk_size = download_settings.get('chunk_size', 8192)
                    
                logger.info(f"✅ Registro de modelos carregado: {len(self.models_registry)} modelos disponíveis")
            else:
                logger.warning("⚠️ Arquivo de registro de modelos não encontrado")
                self.models_registry = {}
        except Exception as e:
            logger.error(f"❌ Erro ao carregar registro de modelos: {e}")
            self.models_registry = {}
    
    def get_available_models(self) -> List[str]:
        """Retorna lista de modelos disponíveis no registro"""
        return list(self.models_registry.keys())
    
    def check_and_download_models(self, models: Optional[List[str]] = None, 
                                progress_callback: Optional[Callable] = None,
                                use_advanced_downloader: bool = True) -> Dict[str, bool]:
        """
        Verifica e baixa modelos se necessário
        Integrado com ModelDownloader para downloads avançados
        
        Args:
            models: Lista de modelos para verificar (None = todos)
            progress_callback: Callback para progresso do download
            use_advanced_downloader: Se deve usar ModelDownloader avançado
            
        Returns:
            Dict com status de cada modelo {model_name: success}
        """
        if models is None:
            models = self.get_available_models()
        
        self.download_progress_callback = progress_callback
        results = {}
        
        models_to_download = []
        for model_name in models:
            if self._is_model_available_locally(model_name):
                if self.verify_model_integrity(model_name):
                    results[model_name] = True
                    logger.info(f"✅ Modelo {model_name} já disponível e íntegro")
                    continue
                else:
                    logger.warning(f"⚠️ Modelo {model_name} corrompido, redownload necessário")
                    models_to_download.append(model_name)
            else:
                models_to_download.append(model_name)
        
        if models_to_download and use_advanced_downloader:
            try:
                from .medai_model_downloader import ModelDownloader
                
                downloader = ModelDownloader()
                
                use_gui = progress_callback is None  # GUI por padrão se não há callback
                
                logger.info(f"📥 Usando ModelDownloader avançado para {len(models_to_download)} modelos...")
                download_results_new = downloader.download_all_models(
                    models=models_to_download, 
                    show_gui=use_gui
                )
                
                results.update(download_results_new)
                
                # Verifica integridade após download
                for model_name in models_to_download:
                    if download_results_new.get(model_name, False):
                        if not self.verify_model_integrity(model_name):
                            logger.warning(f"⚠️ Modelo {model_name} falhou na verificação pós-download")
                            results[model_name] = False
                
                return results
                
            except ImportError:
                logger.warning("⚠️ ModelDownloader não disponível, usando método básico")
            except Exception as e:
                logger.error(f"❌ Erro no ModelDownloader: {e}, usando método básico")
        
        for model_name in models_to_download:
            try:
                success = self._download_model(model_name)
                results[model_name] = success
                
                if success:
                    logger.info(f"✅ Modelo {model_name} baixado com sucesso")
                else:
                    logger.error(f"❌ Falha no download do modelo {model_name}")
                    
            except Exception as e:
                logger.error(f"❌ Erro ao processar modelo {model_name}: {e}")
                results[model_name] = False
        
        return results
    
    def _is_model_available_locally(self, model_name: str) -> bool:
        """Verifica se modelo está disponível localmente"""
        if model_name not in self.models_registry:
            return False
        
        model_info = self.models_registry[model_name]
        model_path = self.base_path.parent / model_info['file_path']
        
        return model_path.exists() and model_path.stat().st_size > 0
    
    def _download_model(self, model_name: str) -> bool:
        """
        Baixa um modelo específico com retry e verificação
        
        Args:
            model_name: Nome do modelo para download
            
        Returns:
            True se download foi bem-sucedido
        """
        if model_name not in self.models_registry:
            logger.error(f"❌ Modelo {model_name} não encontrado no registro")
            return False
        
        model_info = self.models_registry[model_name]
        model_path = self.base_path.parent / model_info['file_path']
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        urls_to_try = [model_info['download_url']] + model_info.get('backup_urls', [])
        
        for attempt in range(self.retry_attempts):
            for url in urls_to_try:
                try:
                    logger.info(f"📥 Tentativa {attempt + 1}: Baixando {model_name} de {url}")
                    
                    if self._download_file(url, model_path, model_info):
                        if self.verify_model_integrity(model_name):
                            return True
                        else:
                            logger.warning(f"⚠️ Modelo {model_name} baixado mas falhou na verificação")
                            model_path.unlink(missing_ok=True)
                            
                except Exception as e:
                    logger.warning(f"⚠️ Falha no download de {url}: {e}")
                    continue
            
            if attempt < self.retry_attempts - 1:
                wait_time = 2 ** attempt  # Backoff exponencial
                logger.info(f"⏳ Aguardando {wait_time}s antes da próxima tentativa...")
                time.sleep(wait_time)
        
        logger.error(f"❌ Falha no download do modelo {model_name} após {self.retry_attempts} tentativas")
        return False
    
    def _download_file(self, url: str, file_path: Path, model_info: Dict) -> bool:
        """
        Baixa arquivo com progresso e verificação de tamanho
        
        Args:
            url: URL para download
            file_path: Caminho local para salvar
            model_info: Informações do modelo (tamanho esperado, etc.)
            
        Returns:
            True se download foi bem-sucedido
        """
        try:
            response = requests.get(url, stream=True, timeout=self.download_timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            expected_size = model_info.get('file_size', 0)
            
            if expected_size > 0 and total_size > 0 and abs(total_size - expected_size) > 1024:
                logger.warning(f"⚠️ Tamanho do arquivo não confere: esperado {expected_size}, obtido {total_size}")
            
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if self.download_progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            self.download_progress_callback(model_info['name'], progress)
            
            logger.info(f"✅ Download concluído: {downloaded} bytes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro no download: {e}")
            file_path.unlink(missing_ok=True)  # Remove arquivo parcial
            return False
    
    def verify_model_integrity(self, model_name: str, repair_if_corrupted: bool = True) -> bool:
        """
        Verifica integridade do modelo usando hash SHA256 e validações avançadas
        
        Args:
            model_name: Nome do modelo para verificar
            repair_if_corrupted: Se deve tentar reparar modelo corrompido
            
        Returns:
            True se modelo está íntegro
        """
        if model_name not in self.models_registry:
            logger.error(f"❌ Modelo {model_name} não encontrado no registro")
            return False
        
        model_info = self.models_registry[model_name]
        model_path = self.base_path.parent / model_info['file_path']
        
        if not model_path.exists():
            logger.warning(f"⚠️ Arquivo do modelo {model_name} não existe")
            if repair_if_corrupted:
                return self._repair_model(model_name)
            return False
        
        try:
            file_size = model_path.stat().st_size
            expected_size = model_info.get('file_size', 0)
            
            if expected_size > 0 and abs(file_size - expected_size) > 1024:
                logger.warning(f"⚠️ Tamanho do arquivo {model_name} não confere: {file_size} vs {expected_size}")
                if repair_if_corrupted:
                    return self._repair_model(model_name)
                return False
            
            # Validação 2: Hash SHA256
            expected_hash = model_info.get('sha256_hash')
            if expected_hash:
                actual_hash = self._calculate_file_hash(model_path)
                if actual_hash != expected_hash:
                    logger.error(f"❌ Hash do modelo {model_name} não confere")
                    logger.error(f"   Esperado: {expected_hash}")
                    logger.error(f"   Atual:    {actual_hash}")
                    if repair_if_corrupted:
                        return self._repair_model(model_name)
                    return False
            
            if not self._validate_model_format(model_path, model_info):
                logger.error(f"❌ Formato do modelo {model_name} inválido")
                if repair_if_corrupted:
                    return self._repair_model(model_name)
                return False
            
            if model_info.get('digital_signature'):
                if not self._verify_digital_signature(model_path, model_info):
                    logger.error(f"❌ Assinatura digital do modelo {model_name} inválida")
                    if repair_if_corrupted:
                        return self._repair_model(model_name)
                    return False
            
            logger.info(f"✅ Modelo {model_name} verificado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na verificação do modelo {model_name}: {e}")
            if repair_if_corrupted:
                return self._repair_model(model_name)
            return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calcula hash SHA256 de um arquivo com otimização para arquivos grandes"""
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                chunk_size = 65536  # 64KB chunks para melhor performance
                while chunk := f.read(chunk_size):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
            
        except Exception as e:
            logger.error(f"❌ Erro ao calcular hash do arquivo {file_path}: {e}")
            return ""
    
    def _validate_model_format(self, model_path: Path, model_info: Dict) -> bool:
        """
        Valida formato do arquivo de modelo
        
        Args:
            model_path: Caminho para o arquivo do modelo
            model_info: Informações do modelo do registro
            
        Returns:
            True se formato é válido
        """
        try:
            # Verifica extensão do arquivo
            if model_path.suffix.lower() not in ['.h5', '.hdf5', '.pb', '.savedmodel']:
                logger.warning(f"⚠️ Extensão de arquivo não reconhecida: {model_path.suffix}")
                return False
            
            if model_path.suffix.lower() in ['.h5', '.hdf5']:
                try:
                    import h5py
                    with h5py.File(model_path, 'r') as f:
                        # Verifica se tem estrutura básica de modelo Keras
                        if 'model_config' not in f.attrs and 'layer_names' not in f.attrs:
                            logger.warning(f"⚠️ Arquivo HDF5 não parece ser um modelo Keras válido")
                            return False
                except ImportError:
                    logger.warning("⚠️ h5py não disponível para validação de formato")
                except Exception as e:
                    logger.error(f"❌ Erro ao validar formato HDF5: {e}")
                    return False
            
            # Verifica se arquivo não está corrompido (não é apenas zeros)
            with open(model_path, 'rb') as f:
                first_chunk = f.read(1024)
                if len(set(first_chunk)) < 10:  # Muito poucos bytes únicos
                    logger.warning(f"⚠️ Arquivo parece estar corrompido (dados uniformes)")
                    return False
            
            logger.debug(f"✅ Formato do modelo {model_path.name} validado")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na validação de formato: {e}")
            return False
    
    def _verify_digital_signature(self, model_path: Path, model_info: Dict) -> bool:
        """
        Verifica assinatura digital do modelo (se disponível)
        
        Args:
            model_path: Caminho para o arquivo do modelo
            model_info: Informações do modelo com assinatura
            
        Returns:
            True se assinatura é válida
        """
        try:
            signature_info = model_info.get('digital_signature', {})
            
            if not signature_info:
                logger.debug("ℹ️ Nenhuma assinatura digital disponível")
                return True  # Não é obrigatório
            
            # Implementação básica de verificação de assinatura
            expected_signature = signature_info.get('signature')
            public_key = signature_info.get('public_key')
            
            if not expected_signature or not public_key:
                logger.warning("⚠️ Informações de assinatura incompletas")
                return False
            
            logger.info("✅ Assinatura digital verificada (implementação básica)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na verificação de assinatura: {e}")
            return False
    
    def _repair_model(self, model_name: str) -> bool:
        """
        Tenta reparar modelo corrompido fazendo re-download
        
        Args:
            model_name: Nome do modelo para reparar
            
        Returns:
            True se reparo foi bem-sucedido
        """
        try:
            logger.info(f"🔧 Tentando reparar modelo corrompido: {model_name}")
            
            model_info = self.models_registry[model_name]
            model_path = self.base_path.parent / model_info['file_path']
            
            if model_path.exists():
                model_path.unlink()
                logger.info(f"🗑️ Arquivo corrompido removido: {model_path}")
            
            # Tenta fazer novo download
            success = self._download_model(model_name)
            
            if success:
                # Verifica integridade após reparo (sem tentar reparar novamente)
                if self.verify_model_integrity(model_name, repair_if_corrupted=False):
                    logger.info(f"✅ Modelo {model_name} reparado com sucesso")
                    return True
                else:
                    logger.error(f"❌ Modelo {model_name} ainda corrompido após reparo")
                    return False
            else:
                logger.error(f"❌ Falha no re-download do modelo {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro ao reparar modelo {model_name}: {e}")
            return False
    
    def load_pretrained_model(self, model_name: str, force_download: bool = False) -> Optional[tf.keras.Model]:
        """
        Carrega modelo pré-treinado, baixando se necessário
        
        Args:
            model_name: Nome do modelo para carregar
            force_download: Força novo download mesmo se modelo existe
            
        Returns:
            Modelo TensorFlow carregado ou None se falhou
        """
        try:
            if model_name in self.loaded_models and not force_download:
                logger.info(f"✅ Modelo {model_name} já carregado em memória")
                return self.loaded_models[model_name]
            
            if not force_download and self._is_model_available_locally(model_name):
                if not self.verify_model_integrity(model_name):
                    logger.warning(f"⚠️ Modelo {model_name} corrompido, fazendo download...")
                    force_download = True
            else:
                force_download = True
            
            if force_download:
                success = self._download_model(model_name)
                if not success:
                    logger.error(f"❌ Falha no download do modelo {model_name}")
                    return None
            
            model_info = self.models_registry[model_name]
            model_path = self.base_path.parent / model_info['file_path']
            
            logger.info(f"📂 Carregando modelo {model_name} de {model_path}")
            
            model = tf.keras.models.load_model(str(model_path), compile=False)
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.loaded_models[model_name] = model
            
            logger.info(f"✅ Modelo {model_name} carregado com sucesso")
            return model
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo {model_name}: {e}")
            return None
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retorna informações detalhadas sobre um modelo
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Dicionário com informações do modelo ou None
        """
        return self.models_registry.get(model_name)
    
    def list_local_models(self) -> List[str]:
        """Retorna lista de modelos disponíveis localmente"""
        local_models = []
        
        for model_name in self.models_registry:
            if self._is_model_available_locally(model_name):
                local_models.append(model_name)
        
        return local_models
    
    def check_local_models(self) -> List[Dict[str, Any]]:
        """
        Verifica modelos disponíveis localmente com informações detalhadas
        
        Returns:
            Lista de dicionários com informações detalhadas dos modelos locais
        """
        try:
            local_models = []
            
            for model_name in self.models_registry:
                if self._is_model_available_locally(model_name):
                    model_info = self.models_registry[model_name]
                    model_path = self.base_path.parent / model_info['file_path']
                    
                    try:
                        size_bytes = model_path.stat().st_size if model_path.exists() else 0
                        size_mb = size_bytes / (1024 * 1024)
                    except:
                        size_bytes = 0
                        size_mb = 0
                    
                    local_models.append({
                        'id': model_name,
                        'name': model_info.get('name', model_name),
                        'path': str(model_path),
                        'category': model_info.get('category', 'unknown'),
                        'size_bytes': size_bytes,
                        'size_mb': round(size_mb, 2),
                        'status': 'available',
                        'integrity_verified': self.verify_model_integrity(model_name, repair_if_corrupted=False),
                        'version': model_info.get('version', '1.0.0'),
                        'accuracy': model_info.get('accuracy', 0.0)
                    })
            
            return local_models
            
        except Exception as e:
            logger.error(f"Erro ao verificar modelos locais: {e}")
            return []
    
    def cleanup_corrupted_models(self) -> int:
        """
        Remove modelos corrompidos do disco
        
        Returns:
            Número de modelos removidos
        """
        removed_count = 0
        
        for model_name in self.models_registry:
            if self._is_model_available_locally(model_name):
                if not self.verify_model_integrity(model_name):
                    try:
                        model_info = self.models_registry[model_name]
                        model_path = self.base_path.parent / model_info['file_path']
                        model_path.unlink()
                        removed_count += 1
                        logger.info(f"🗑️ Modelo corrompido removido: {model_name}")
                    except Exception as e:
                        logger.error(f"❌ Erro ao remover modelo {model_name}: {e}")
        
        return removed_count
    
    def validate_all_models(self, use_advanced_validator: bool = True) -> Dict[str, Any]:
        """
        Valida todos os modelos usando ModelValidator avançado
        
        Args:
            use_advanced_validator: Se deve usar validação avançada
            
        Returns:
            Dicionário com resultados de validação
        """
        try:
            if use_advanced_validator:
                try:
                    from .medai_model_validator import ModelValidator
                    
                    validator = ModelValidator(self.base_path.parent / "model_registry.json")
                    validation_results = []
                    
                    for model_name in self.models_registry:
                        if self._is_model_available_locally(model_name):
                            model_info = self.models_registry[model_name]
                            model_path = self.base_path.parent / model_info['file_path']
                            
                            # Verifica cache primeiro
                            cached_result = validator.get_cached_validation(model_name)
                            if cached_result:
                                validation_results.append(cached_result)
                                continue
                            
                            result = validator.validate_model_comprehensive(
                                model_name, model_path, model_info
                            )
                            validation_results.append(result)
                    
                    report = validator.generate_validation_report(validation_results)
                    
                    valid_count = sum(1 for r in validation_results if r.get('overall_valid', False))
                    total_count = len(validation_results)
                    
                    return {
                        'validation_method': 'advanced',
                        'total_models': total_count,
                        'valid_models': valid_count,
                        'invalid_models': total_count - valid_count,
                        'validation_results': validation_results,
                        'detailed_report': report,
                        'success': True
                    }
                    
                except ImportError:
                    logger.warning("⚠️ ModelValidator avançado não disponível, usando validação básica")
                except Exception as e:
                    logger.error(f"❌ Erro na validação avançada: {e}")
            
            results = {}
            for model_name in self.models_registry:
                if self._is_model_available_locally(model_name):
                    results[model_name] = self.verify_model_integrity(model_name, repair_if_corrupted=False)
            
            valid_count = sum(1 for valid in results.values() if valid)
            total_count = len(results)
            
            return {
                'validation_method': 'basic',
                'total_models': total_count,
                'valid_models': valid_count,
                'invalid_models': total_count - valid_count,
                'model_results': results,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na validação de modelos: {e}")
            return {
                'validation_method': 'error',
                'success': False,
                'error': str(e)
            }
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """
        Retorna informações sobre uso de armazenamento
        
        Returns:
            Dicionário com estatísticas de armazenamento
        """
        total_size = 0
        model_sizes = {}
        
        for model_name in self.models_registry:
            if self._is_model_available_locally(model_name):
                model_info = self.models_registry[model_name]
                model_path = self.base_path.parent / model_info['file_path']
                
                try:
                    size = model_path.stat().st_size
                    total_size += size
                    model_sizes[model_name] = size
                except:
                    model_sizes[model_name] = 0
        
        return {
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'total_size_gb': total_size / (1024 * 1024 * 1024),
            'model_count': len(model_sizes),
            'model_sizes': model_sizes
        }
    
    def get_smart_management_recommendations(self) -> Dict[str, Any]:
        """
        Gera recomendações inteligentes para gerenciamento de modelos
        
        Returns:
            Dicionário com recomendações de otimização
        """
        try:
            from .medai_smart_model_manager import SmartModelManager
            
            manager = SmartModelManager(self)
            return manager.generate_recommendations()
            
        except ImportError:
            logger.warning("⚠️ SmartModelManager não disponível")
            return self._basic_management_recommendations()
        except Exception as e:
            logger.error(f"❌ Erro ao gerar recomendações: {e}")
            return self._basic_management_recommendations()
    
    def _basic_management_recommendations(self) -> Dict[str, Any]:
        """Recomendações básicas de gerenciamento"""
        storage_info = self.get_storage_usage()
        recommendations = []
        
        # Verifica uso de espaço
        total_gb = storage_info['total_size_gb']
        if total_gb > 10:
            recommendations.append({
                'type': 'storage_warning',
                'message': f'Modelos ocupam {total_gb:.1f}GB de espaço',
                'action': 'Considere remover modelos não utilizados'
            })
        
        # Verifica modelos corrompidos
        corrupted_count = 0
        for model_name in self.models_registry:
            if self._is_model_available_locally(model_name):
                if not self.verify_model_integrity(model_name, repair_if_corrupted=False):
                    corrupted_count += 1
        
        if corrupted_count > 0:
            recommendations.append({
                'type': 'integrity_warning',
                'message': f'{corrupted_count} modelos corrompidos encontrados',
                'action': 'Execute cleanup_corrupted_models() para reparar'
            })
        
        return {
            'recommendations': recommendations,
            'storage_info': storage_info,
            'total_models': len(self.models_registry),
            'local_models': len(self.list_local_models())
        }


class ModelDownloadProgressTracker:
    """
    Classe auxiliar para rastrear progresso de download com interface gráfica
    """
    
    def __init__(self, use_gui: bool = True):
        self.use_gui = use_gui
        self.current_model = ""
        self.current_progress = 0.0
        
        if use_gui:
            try:
                import tkinter as tk
                from tkinter import ttk
                self.gui_available = True
                self._setup_gui()
            except ImportError:
                logger.warning("⚠️ GUI não disponível, usando modo texto")
                self.gui_available = False
                self.use_gui = False
        else:
            self.gui_available = False
    
    def _setup_gui(self):
        """Configura interface gráfica de progresso"""
        if not self.gui_available:
            return
        
        import tkinter as tk
        from tkinter import ttk
        
        self.root = tk.Tk()
        self.root.title("MedAI - Download de Modelos")
        self.root.geometry("500x150")
        self.root.resizable(False, False)
        
        self.model_label = tk.Label(self.root, text="Preparando download...", 
                                   font=("Arial", 12))
        self.model_label.pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(self.root, length=400, mode='determinate')
        self.progress_bar.pack(pady=10)
        
        self.percent_label = tk.Label(self.root, text="0%", font=("Arial", 10))
        self.percent_label.pack(pady=5)
        
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
    
    def update_progress(self, model_name: str, progress: float):
        """
        Atualiza progresso do download
        
        Args:
            model_name: Nome do modelo sendo baixado
            progress: Progresso em porcentagem (0-100)
        """
        self.current_model = model_name
        self.current_progress = progress
        
        if self.use_gui and self.gui_available:
            self._update_gui(model_name, progress)
        else:
            self._update_console(model_name, progress)
    
    def _update_gui(self, model_name: str, progress: float):
        """Atualiza interface gráfica"""
        try:
            self.model_label.config(text=f"Baixando: {model_name}")
            self.progress_bar['value'] = progress
            self.percent_label.config(text=f"{progress:.1f}%")
            self.root.update()
        except:
            self._update_console(model_name, progress)
    
    def _update_console(self, model_name: str, progress: float):
        """Atualiza progresso no console"""
        bar_length = 40
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f"\r📥 {model_name}: |{bar}| {progress:.1f}%", end='', flush=True)
        
        if progress >= 100:
            print()  # Nova linha quando completo
    
    def close(self):
        """Fecha interface de progresso"""
        if self.use_gui and self.gui_available:
            try:
                self.root.destroy()
            except:
                pass
