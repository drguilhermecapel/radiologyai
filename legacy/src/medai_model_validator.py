"""
Sistema de Validação de Modelos para MedAI Radiologia
Implementa verificação avançada de integridade, autenticidade e licenças
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import time

logger = logging.getLogger('MedAI.ModelValidator')

class ModelValidator:
    """
    Validador avançado de modelos de IA com verificação de integridade,
    autenticidade e conformidade de licenças
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or Path("models/model_registry.json")
        self.validation_cache = {}
        self.cache_duration = timedelta(hours=24)  # Cache válido por 24h
        
        self.supported_hash_algorithms = {
            'sha256': hashlib.sha256,
            'sha512': hashlib.sha512,
            'md5': hashlib.md5  # Apenas para compatibilidade
        }
        
        self.supported_formats = {
            '.h5': 'Keras HDF5',
            '.hdf5': 'Keras HDF5',
            '.pb': 'TensorFlow SavedModel',
            '.savedmodel': 'TensorFlow SavedModel',
            '.onnx': 'ONNX Model',
            '.pkl': 'Pickle Model',
            '.joblib': 'Joblib Model'
        }
        
        logger.info("✅ ModelValidator inicializado")
    
    def validate_model_comprehensive(self, model_name: str, model_path: Path, 
                                   model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validação abrangente de modelo incluindo todas as verificações
        
        Args:
            model_name: Nome do modelo
            model_path: Caminho para o arquivo do modelo
            model_info: Informações do modelo do registro
            
        Returns:
            Dicionário com resultados detalhados da validação
        """
        validation_start = time.time()
        
        results = {
            'model_name': model_name,
            'validation_timestamp': datetime.now().isoformat(),
            'overall_valid': False,
            'checks': {},
            'warnings': [],
            'errors': [],
            'validation_time_seconds': 0
        }
        
        try:
            logger.info(f"🔍 Iniciando validação abrangente do modelo: {model_name}")
            
            results['checks']['file_exists'] = self._check_file_exists(model_path, results)
            
            if not results['checks']['file_exists']:
                results['overall_valid'] = False
                return results
            
            results['checks']['size_valid'] = self._check_file_size(
                model_path, model_info, results
            )
            
            results['checks']['format_valid'] = self._check_file_format(
                model_path, model_info, results
            )
            
            results['checks']['integrity_valid'] = self._check_file_integrity(
                model_path, model_info, results
            )
            
            results['checks']['signature_valid'] = self._check_digital_signature(
                model_path, model_info, results
            )
            
            results['checks']['license_valid'] = self._check_license_compliance(
                model_info, results
            )
            
            results['checks']['metadata_valid'] = self._check_model_metadata(
                model_path, model_info, results
            )
            
            results['checks']['security_valid'] = self._check_security_compliance(
                model_path, model_info, results
            )
            
            critical_checks = ['file_exists', 'integrity_valid', 'license_valid']
            results['overall_valid'] = all(
                results['checks'].get(check, False) for check in critical_checks
            )
            
            validation_time = time.time() - validation_start
            results['validation_time_seconds'] = round(validation_time, 3)
            
            if results['overall_valid']:
                logger.info(f"✅ Modelo {model_name} passou em todas as validações críticas")
            else:
                logger.warning(f"⚠️ Modelo {model_name} falhou em validações críticas")
            
            self._update_validation_cache(model_name, results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erro na validação abrangente do modelo {model_name}: {e}")
            results['errors'].append(f"Erro geral na validação: {str(e)}")
            results['overall_valid'] = False
            return results
    
    def _check_file_exists(self, model_path: Path, results: Dict) -> bool:
        """Verifica se arquivo do modelo existe"""
        try:
            exists = model_path.exists() and model_path.is_file()
            if not exists:
                results['errors'].append(f"Arquivo do modelo não encontrado: {model_path}")
            return exists
        except Exception as e:
            results['errors'].append(f"Erro ao verificar existência do arquivo: {e}")
            return False
    
    def _check_file_size(self, model_path: Path, model_info: Dict, results: Dict) -> bool:
        """Verifica se tamanho do arquivo está correto"""
        try:
            actual_size = model_path.stat().st_size
            expected_size = model_info.get('file_size', 0)
            
            if expected_size == 0:
                results['warnings'].append("Tamanho esperado não especificado no registro")
                return True  # Não é crítico se não especificado
            
            size_diff = abs(actual_size - expected_size)
            tolerance = 1024
            
            if size_diff > tolerance:
                results['errors'].append(
                    f"Tamanho do arquivo incorreto: {actual_size} bytes "
                    f"(esperado: {expected_size} bytes, diferença: {size_diff} bytes)"
                )
                return False
            
            if size_diff > 0:
                results['warnings'].append(
                    f"Pequena diferença no tamanho: {size_diff} bytes (dentro da tolerância)"
                )
            
            return True
            
        except Exception as e:
            results['errors'].append(f"Erro ao verificar tamanho do arquivo: {e}")
            return False
    
    def _check_file_format(self, model_path: Path, model_info: Dict, results: Dict) -> bool:
        """Verifica se formato do arquivo é válido"""
        try:
            file_extension = model_path.suffix.lower()
            
            if file_extension not in self.supported_formats:
                results['errors'].append(f"Formato de arquivo não suportado: {file_extension}")
                return False
            
            if file_extension in ['.h5', '.hdf5']:
                return self._validate_hdf5_format(model_path, results)
            elif file_extension == '.pb':
                return self._validate_tensorflow_format(model_path, results)
            elif file_extension == '.onnx':
                return self._validate_onnx_format(model_path, results)
            
            return True
            
        except Exception as e:
            results['errors'].append(f"Erro ao verificar formato do arquivo: {e}")
            return False
    
    def _validate_hdf5_format(self, model_path: Path, results: Dict) -> bool:
        """Valida formato HDF5 (Keras)"""
        try:
            import h5py
            
            with h5py.File(model_path, 'r') as f:
                has_model_config = 'model_config' in f.attrs
                has_layer_names = 'layer_names' in f.attrs
                has_model_weights = 'model_weights' in f
                
                if not (has_model_config or has_layer_names or has_model_weights):
                    results['warnings'].append(
                        "Arquivo HDF5 pode não ser um modelo Keras válido"
                    )
                    return False
                
                return True
                
        except ImportError:
            results['warnings'].append("h5py não disponível para validação HDF5")
            return True  # Não falha se biblioteca não disponível
        except Exception as e:
            results['errors'].append(f"Erro na validação HDF5: {e}")
            return False
    
    def _validate_tensorflow_format(self, model_path: Path, results: Dict) -> bool:
        """Valida formato TensorFlow"""
        try:
            with open(model_path, 'rb') as f:
                header = f.read(8)
                if len(header) < 8:
                    results['errors'].append("Arquivo .pb muito pequeno")
                    return False
            
            return True
            
        except Exception as e:
            results['errors'].append(f"Erro na validação TensorFlow: {e}")
            return False
    
    def _validate_onnx_format(self, model_path: Path, results: Dict) -> bool:
        """Valida formato ONNX"""
        try:
            import onnx
            
            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)
            return True
            
        except ImportError:
            results['warnings'].append("onnx não disponível para validação ONNX")
            return True
        except Exception as e:
            results['errors'].append(f"Erro na validação ONNX: {e}")
            return False
    
    def _check_file_integrity(self, model_path: Path, model_info: Dict, results: Dict) -> bool:
        """Verifica integridade usando hash"""
        try:
            hash_info = model_info.get('sha256_hash')
            algorithm = 'sha256'
            
            if not hash_info:
                for alg in ['sha512', 'md5']:
                    if f'{alg}_hash' in model_info:
                        hash_info = model_info[f'{alg}_hash']
                        algorithm = alg
                        break
            
            if not hash_info:
                results['warnings'].append("Nenhum hash disponível para verificação")
                return True  # Não é crítico se não especificado
            
            actual_hash = self._calculate_file_hash(model_path, algorithm)
            
            if actual_hash.lower() != hash_info.lower():
                results['errors'].append(
                    f"Hash {algorithm.upper()} não confere:\n"
                    f"  Esperado: {hash_info}\n"
                    f"  Atual:    {actual_hash}"
                )
                return False
            
            logger.debug(f"✅ Hash {algorithm.upper()} verificado com sucesso")
            return True
            
        except Exception as e:
            results['errors'].append(f"Erro na verificação de integridade: {e}")
            return False
    
    def _calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calcula hash do arquivo usando algoritmo especificado"""
        if algorithm not in self.supported_hash_algorithms:
            raise ValueError(f"Algoritmo de hash não suportado: {algorithm}")
        
        hash_func = self.supported_hash_algorithms[algorithm]()
        
        with open(file_path, 'rb') as f:
            chunk_size = 65536  # 64KB chunks
            while chunk := f.read(chunk_size):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def _check_digital_signature(self, model_path: Path, model_info: Dict, results: Dict) -> bool:
        """Verifica assinatura digital do modelo"""
        try:
            signature_info = model_info.get('digital_signature')
            
            if not signature_info:
                results['warnings'].append("Nenhuma assinatura digital disponível")
                return True  # Não é obrigatório
            
            signature = signature_info.get('signature')
            public_key = signature_info.get('public_key')
            algorithm = signature_info.get('algorithm', 'RSA-SHA256')
            
            if not signature or not public_key:
                results['errors'].append("Informações de assinatura incompletas")
                return False
            
            logger.debug("ℹ️ Verificação de assinatura digital (implementação básica)")
            results['warnings'].append("Verificação de assinatura digital não implementada completamente")
            
            return True
            
        except Exception as e:
            results['errors'].append(f"Erro na verificação de assinatura: {e}")
            return False
    
    def _check_license_compliance(self, model_info: Dict, results: Dict) -> bool:
        """Verifica conformidade de licença"""
        try:
            license_info = model_info.get('license')
            
            if not license_info:
                results['errors'].append("Informações de licença não disponíveis")
                return False
            
            accepted_licenses = [
                'Apache-2.0', 'MIT', 'BSD-3-Clause', 'BSD-2-Clause',
                'GPL-3.0', 'LGPL-3.0', 'CC-BY-4.0', 'CC-BY-SA-4.0'
            ]
            
            if license_info not in accepted_licenses:
                results['warnings'].append(
                    f"Licença '{license_info}' pode não ser adequada para uso médico"
                )
            
            restrictions = model_info.get('license_restrictions', [])
            if 'commercial_use_prohibited' in restrictions:
                results['warnings'].append("Uso comercial pode ser restrito pela licença")
            
            if 'medical_use_requires_approval' in restrictions:
                results['warnings'].append("Uso médico pode requerer aprovação adicional")
            
            return True
            
        except Exception as e:
            results['errors'].append(f"Erro na verificação de licença: {e}")
            return False
    
    def _check_model_metadata(self, model_path: Path, model_info: Dict, results: Dict) -> bool:
        """Verifica metadados do modelo"""
        try:
            required_metadata = [
                'name', 'version', 'architecture', 'input_shape', 
                'classes', 'accuracy', 'clinical_validation'
            ]
            
            missing_metadata = []
            for field in required_metadata:
                if field not in model_info:
                    missing_metadata.append(field)
            
            if missing_metadata:
                results['warnings'].append(
                    f"Metadados ausentes: {', '.join(missing_metadata)}"
                )
            
            clinical_validation = model_info.get('clinical_validation', {})
            if not clinical_validation:
                results['warnings'].append("Informações de validação clínica não disponíveis")
            else:
                validation_date = clinical_validation.get('validation_date')
                if validation_date:
                    try:
                        val_date = datetime.fromisoformat(validation_date.replace('Z', '+00:00'))
                        age = datetime.now() - val_date.replace(tzinfo=None)
                        
                        if age > timedelta(days=365 * 2):  # 2 anos
                            results['warnings'].append(
                                f"Validação clínica antiga ({age.days} dias)"
                            )
                    except:
                        results['warnings'].append("Data de validação clínica inválida")
            
            return True
            
        except Exception as e:
            results['errors'].append(f"Erro na verificação de metadados: {e}")
            return False
    
    def _check_security_compliance(self, model_path: Path, model_info: Dict, results: Dict) -> bool:
        """Verifica conformidade de segurança"""
        try:
            file_size = model_path.stat().st_size
            
            if file_size < 1024:  # 1KB
                results['warnings'].append("Arquivo de modelo muito pequeno, pode estar corrompido")
            
            if file_size > 10 * 1024 * 1024 * 1024:  # 10GB
                results['warnings'].append("Arquivo de modelo muito grande, verificar se é legítimo")
            
            if model_path.stat().st_mode & 0o111:  # Executável
                results['warnings'].append("Arquivo de modelo tem permissões de execução")
            
            download_url = model_info.get('download_url', '')
            if download_url:
                trusted_domains = [
                    'models.medai.com', 'huggingface.co', 'tensorflow.org',
                    'pytorch.org', 'github.com', 'gitlab.com'
                ]
                
                is_trusted = any(domain in download_url for domain in trusted_domains)
                if not is_trusted:
                    results['warnings'].append(f"Origem do modelo não está na lista de confiáveis: {download_url}")
            
            return True
            
        except Exception as e:
            results['errors'].append(f"Erro na verificação de segurança: {e}")
            return False
    
    def _update_validation_cache(self, model_name: str, results: Dict):
        """Atualiza cache de validação"""
        try:
            self.validation_cache[model_name] = {
                'results': results,
                'timestamp': datetime.now(),
                'expires_at': datetime.now() + self.cache_duration
            }
        except Exception as e:
            logger.warning(f"⚠️ Erro ao atualizar cache de validação: {e}")
    
    def get_cached_validation(self, model_name: str) -> Optional[Dict]:
        """Retorna validação em cache se ainda válida"""
        try:
            cached = self.validation_cache.get(model_name)
            if cached and datetime.now() < cached['expires_at']:
                logger.debug(f"✅ Usando validação em cache para {model_name}")
                return cached['results']
            return None
        except Exception:
            return None
    
    def generate_validation_report(self, validation_results: List[Dict]) -> str:
        """Gera relatório de validação em formato texto"""
        try:
            report_lines = [
                "=" * 80,
                "RELATÓRIO DE VALIDAÇÃO DE MODELOS - MedAI Radiologia",
                "=" * 80,
                f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Total de modelos validados: {len(validation_results)}",
                ""
            ]
            
            valid_count = sum(1 for r in validation_results if r.get('overall_valid', False))
            invalid_count = len(validation_results) - valid_count
            
            report_lines.extend([
                f"✅ Modelos válidos: {valid_count}",
                f"❌ Modelos inválidos: {invalid_count}",
                ""
            ])
            
            for result in validation_results:
                model_name = result.get('model_name', 'Desconhecido')
                overall_valid = result.get('overall_valid', False)
                validation_time = result.get('validation_time_seconds', 0)
                
                status = "✅ VÁLIDO" if overall_valid else "❌ INVÁLIDO"
                
                report_lines.extend([
                    f"Modelo: {model_name}",
                    f"Status: {status}",
                    f"Tempo de validação: {validation_time}s",
                ])
                
                errors = result.get('errors', [])
                if errors:
                    report_lines.append("Erros:")
                    for error in errors:
                        report_lines.append(f"  - {error}")
                
                warnings = result.get('warnings', [])
                if warnings:
                    report_lines.append("Avisos:")
                    for warning in warnings:
                        report_lines.append(f"  - {warning}")
                
                report_lines.append("-" * 40)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar relatório de validação: {e}")
            return f"Erro ao gerar relatório: {e}"
