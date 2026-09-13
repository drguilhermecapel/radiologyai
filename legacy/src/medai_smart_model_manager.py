"""
Sistema de Gerenciamento Inteligente de Modelos para MedAI Radiologia
Implementa otimização automática, cache inteligente e recomendações de uso
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import threading

logger = logging.getLogger('MedAI.SmartModelManager')

class SmartModelManager:
    """
    Gerenciador inteligente de modelos com otimização automática,
    cache adaptativo e recomendações baseadas em uso
    """
    
    def __init__(self, pretrained_loader):
        self.pretrained_loader = pretrained_loader
        self.usage_stats_file = Path("models/usage_stats.json")
        self.cache_policy_file = Path("models/cache_policy.json")
        
        self.usage_stats = self._load_usage_stats()
        self.cache_policy = self._load_cache_policy()
        
        self.optimization_config = {
            'max_cache_size_gb': 20,
            'min_free_space_gb': 5,
            'auto_cleanup_enabled': True,
            'usage_tracking_enabled': True,
            'preload_popular_models': True,
            'adaptive_caching': True
        }
        
        self.performance_metrics = {
            'load_times': defaultdict(list),
            'memory_usage': defaultdict(list),
            'accuracy_scores': defaultdict(list),
            'user_satisfaction': defaultdict(list)
        }
        
        logger.info("✅ SmartModelManager inicializado")
    
    def track_model_usage(self, model_name: str, operation: str, 
                         duration: float = 0, success: bool = True):
        """
        Rastreia uso de modelo para otimização futura
        
        Args:
            model_name: Nome do modelo usado
            operation: Tipo de operação (load, predict, validate)
            duration: Tempo gasto na operação
            success: Se operação foi bem-sucedida
        """
        if not self.optimization_config['usage_tracking_enabled']:
            return
        
        try:
            timestamp = datetime.now().isoformat()
            
            if model_name not in self.usage_stats:
                self.usage_stats[model_name] = {
                    'total_uses': 0,
                    'last_used': timestamp,
                    'operations': defaultdict(int),
                    'avg_load_time': 0,
                    'success_rate': 1.0,
                    'user_rating': 5.0,
                    'priority_score': 0
                }
            
            stats = self.usage_stats[model_name]
            stats['total_uses'] += 1
            stats['last_used'] = timestamp
            stats['operations'][operation] += 1
            
            if operation == 'load' and duration > 0:
                load_times = self.performance_metrics['load_times'][model_name]
                load_times.append(duration)
                
                if len(load_times) > 50:
                    load_times.pop(0)
                
                stats['avg_load_time'] = sum(load_times) / len(load_times)
            
            total_ops = sum(stats['operations'].values())
            if not success:
                current_successes = stats['success_rate'] * (total_ops - 1)
                stats['success_rate'] = current_successes / total_ops
            
            stats['priority_score'] = self._calculate_priority_score(stats)
            
            if stats['total_uses'] % 10 == 0:
                self._save_usage_stats()
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao rastrear uso do modelo {model_name}: {e}")
    
    def _calculate_priority_score(self, stats: Dict) -> float:
        """Calcula score de prioridade baseado em múltricos fatores"""
        try:
            usage_factor = min(stats['total_uses'] / 100, 1.0)  # Normalizado para 100 usos
            recency_factor = self._calculate_recency_factor(stats['last_used'])
            performance_factor = min(stats['success_rate'], 1.0)
            speed_factor = max(0, 1.0 - (stats['avg_load_time'] / 60))  # Penaliza modelos lentos
            rating_factor = stats['user_rating'] / 5.0
            
            weights = {
                'usage': 0.3,
                'recency': 0.2,
                'performance': 0.2,
                'speed': 0.15,
                'rating': 0.15
            }
            
            priority_score = (
                usage_factor * weights['usage'] +
                recency_factor * weights['recency'] +
                performance_factor * weights['performance'] +
                speed_factor * weights['speed'] +
                rating_factor * weights['rating']
            )
            
            return round(priority_score, 3)
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao calcular score de prioridade: {e}")
            return 0.5  # Score neutro
    
    def _calculate_recency_factor(self, last_used_str: str) -> float:
        """Calcula fator de recência (mais recente = maior score)"""
        try:
            last_used = datetime.fromisoformat(last_used_str.replace('Z', '+00:00'))
            last_used = last_used.replace(tzinfo=None)
            
            days_ago = (datetime.now() - last_used).days
            
            if days_ago == 0:
                return 1.0
            elif days_ago <= 7:
                return 0.8
            elif days_ago <= 30:
                return 0.5
            elif days_ago <= 90:
                return 0.2
            else:
                return 0.1
                
        except Exception:
            return 0.5  # Score neutro se não conseguir calcular
    
    def generate_recommendations(self) -> Dict[str, Any]:
        """
        Gera recomendações inteligentes de otimização
        
        Returns:
            Dicionário com recomendações detalhadas
        """
        try:
            recommendations = {
                'cache_optimization': self._analyze_cache_optimization(),
                'storage_optimization': self._analyze_storage_optimization(),
                'performance_optimization': self._analyze_performance_optimization(),
                'usage_insights': self._analyze_usage_patterns(),
                'maintenance_tasks': self._generate_maintenance_tasks(),
                'priority_models': self._get_priority_models(),
                'generated_at': datetime.now().isoformat()
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar recomendações: {e}")
            return {'error': str(e)}
    
    def _analyze_cache_optimization(self) -> Dict[str, Any]:
        """Analisa otimizações de cache"""
        try:
            storage_info = self.pretrained_loader.get_storage_usage()
            current_size_gb = storage_info['total_size_gb']
            max_cache_gb = self.optimization_config['max_cache_size_gb']
            
            recommendations = []
            
            if current_size_gb > max_cache_gb:
                models_to_remove = self._identify_models_for_removal()
                recommendations.append({
                    'type': 'cache_cleanup',
                    'priority': 'high',
                    'message': f'Cache excede limite ({current_size_gb:.1f}GB > {max_cache_gb}GB)',
                    'action': f'Remover {len(models_to_remove)} modelos menos usados',
                    'models_to_remove': models_to_remove[:5]  # Top 5
                })
            
            if self.optimization_config['preload_popular_models']:
                popular_models = self._get_popular_models_not_cached()
                if popular_models:
                    recommendations.append({
                        'type': 'preload_models',
                        'priority': 'medium',
                        'message': f'{len(popular_models)} modelos populares não estão em cache',
                        'action': 'Pré-carregar modelos frequentemente usados',
                        'models_to_preload': popular_models[:3]  # Top 3
                    })
            
            return {
                'current_cache_size_gb': current_size_gb,
                'max_cache_size_gb': max_cache_gb,
                'cache_utilization': min(current_size_gb / max_cache_gb, 1.0),
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na análise de cache: {e}")
            return {'error': str(e)}
    
    def _analyze_storage_optimization(self) -> Dict[str, Any]:
        """Analisa otimizações de armazenamento"""
        try:
            storage_info = self.pretrained_loader.get_storage_usage()
            recommendations = []
            
            duplicates = self._find_duplicate_models()
            if duplicates:
                recommendations.append({
                    'type': 'remove_duplicates',
                    'priority': 'medium',
                    'message': f'{len(duplicates)} modelos duplicados encontrados',
                    'action': 'Remover versões antigas ou redundantes',
                    'duplicate_models': duplicates
                })
            
            unused_models = self._find_unused_models()
            if unused_models:
                space_saved_gb = sum(
                    storage_info['model_sizes'].get(model, 0) 
                    for model in unused_models
                ) / (1024**3)
                
                recommendations.append({
                    'type': 'remove_unused',
                    'priority': 'low',
                    'message': f'{len(unused_models)} modelos não utilizados',
                    'action': f'Remover para economizar {space_saved_gb:.1f}GB',
                    'unused_models': unused_models[:5]
                })
            
            return {
                'total_models': storage_info['model_count'],
                'total_size_gb': storage_info['total_size_gb'],
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na análise de armazenamento: {e}")
            return {'error': str(e)}
    
    def _analyze_performance_optimization(self) -> Dict[str, Any]:
        """Analisa otimizações de performance"""
        try:
            recommendations = []
            
            slow_models = []
            for model_name, load_times in self.performance_metrics['load_times'].items():
                if load_times:
                    avg_time = sum(load_times) / len(load_times)
                    if avg_time > 30:  # Mais de 30 segundos
                        slow_models.append({
                            'model': model_name,
                            'avg_load_time': avg_time,
                            'suggestion': 'Considere otimização ou modelo alternativo'
                        })
            
            if slow_models:
                recommendations.append({
                    'type': 'optimize_slow_models',
                    'priority': 'medium',
                    'message': f'{len(slow_models)} modelos com carregamento lento',
                    'action': 'Otimizar ou substituir modelos lentos',
                    'slow_models': slow_models[:3]
                })
            
            unreliable_models = []
            for model_name, stats in self.usage_stats.items():
                if stats['success_rate'] < 0.9 and stats['total_uses'] > 5:
                    unreliable_models.append({
                        'model': model_name,
                        'success_rate': stats['success_rate'],
                        'total_uses': stats['total_uses']
                    })
            
            if unreliable_models:
                recommendations.append({
                    'type': 'fix_unreliable_models',
                    'priority': 'high',
                    'message': f'{len(unreliable_models)} modelos com baixa confiabilidade',
                    'action': 'Investigar e corrigir problemas',
                    'unreliable_models': unreliable_models
                })
            
            return {
                'total_tracked_models': len(self.performance_metrics['load_times']),
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na análise de performance: {e}")
            return {'error': str(e)}
    
    def _analyze_usage_patterns(self) -> Dict[str, Any]:
        """Analisa padrões de uso dos modelos"""
        try:
            if not self.usage_stats:
                return {'message': 'Dados de uso insuficientes'}
            
            most_used = sorted(
                self.usage_stats.items(),
                key=lambda x: x[1]['total_uses'],
                reverse=True
            )[:5]
            
            most_recent = sorted(
                self.usage_stats.items(),
                key=lambda x: x[1]['last_used'],
                reverse=True
            )[:5]
            
            usage_by_hour = defaultdict(int)
            for model_name, stats in self.usage_stats.items():
                try:
                    last_used = datetime.fromisoformat(stats['last_used'].replace('Z', '+00:00'))
                    hour = last_used.hour
                    usage_by_hour[hour] += stats['total_uses']
                except:
                    continue
            
            peak_hours = sorted(usage_by_hour.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                'most_used_models': [{'model': name, 'uses': stats['total_uses']} 
                                   for name, stats in most_used],
                'recently_used_models': [{'model': name, 'last_used': stats['last_used']} 
                                       for name, stats in most_recent],
                'peak_usage_hours': [{'hour': hour, 'usage': count} 
                                   for hour, count in peak_hours],
                'total_tracked_models': len(self.usage_stats)
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na análise de padrões: {e}")
            return {'error': str(e)}
    
    def _generate_maintenance_tasks(self) -> List[Dict[str, Any]]:
        """Gera tarefas de manutenção recomendadas"""
        tasks = []
        
        try:
            tasks.append({
                'task': 'integrity_check',
                'priority': 'high',
                'description': 'Verificar integridade de todos os modelos',
                'estimated_time': '5-10 minutos',
                'command': 'pretrained_loader.validate_all_models()'
            })
            
            storage_info = self.pretrained_loader.get_storage_usage()
            if storage_info['total_size_gb'] > 15:
                tasks.append({
                    'task': 'cache_cleanup',
                    'priority': 'medium',
                    'description': 'Limpar modelos não utilizados do cache',
                    'estimated_time': '2-5 minutos',
                    'command': 'smart_manager.cleanup_unused_models()'
                })
            
            tasks.append({
                'task': 'model_updates',
                'priority': 'low',
                'description': 'Verificar atualizações disponíveis para modelos',
                'estimated_time': '10-15 minutos',
                'command': 'pretrained_loader.check_for_updates()'
            })
            
            tasks.append({
                'task': 'backup_stats',
                'priority': 'low',
                'description': 'Fazer backup das estatísticas de uso',
                'estimated_time': '1 minuto',
                'command': 'smart_manager.backup_usage_stats()'
            })
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar tarefas de manutenção: {e}")
        
        return tasks
    
    def _get_priority_models(self) -> List[Dict[str, Any]]:
        """Retorna modelos ordenados por prioridade"""
        try:
            priority_models = []
            
            for model_name, stats in self.usage_stats.items():
                priority_models.append({
                    'model': model_name,
                    'priority_score': stats['priority_score'],
                    'total_uses': stats['total_uses'],
                    'success_rate': stats['success_rate'],
                    'avg_load_time': stats['avg_load_time'],
                    'last_used': stats['last_used']
                })
            
            priority_models.sort(key=lambda x: x['priority_score'], reverse=True)
            
            return priority_models[:10]  # Top 10
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter modelos prioritários: {e}")
            return []
    
    def _identify_models_for_removal(self) -> List[str]:
        """Identifica modelos candidatos para remoção"""
        candidates = []
        
        for model_name, stats in self.usage_stats.items():
            
            if (stats['priority_score'] < 0.3 or
                stats['success_rate'] < 0.7 or
                self._days_since_last_use(stats['last_used']) > 90):
                
                candidates.append(model_name)
        
        return candidates
    
    def _get_popular_models_not_cached(self) -> List[str]:
        """Identifica modelos populares que não estão em cache"""
        popular_not_cached = []
        
        for model_name, stats in self.usage_stats.items():
            if (stats['priority_score'] > 0.7 and
                not self.pretrained_loader._is_model_available_locally(model_name)):
                popular_not_cached.append(model_name)
        
        return popular_not_cached
    
    def _find_duplicate_models(self) -> List[Dict[str, Any]]:
        """Encontra modelos duplicados ou redundantes"""
        duplicates = []
        
        model_groups = defaultdict(list)
        for model_name in self.pretrained_loader.models_registry:
            base_name = model_name.split('_')[0]  # ex: "chest_xray" de "chest_xray_efficientnetv2"
            model_groups[base_name].append(model_name)
        
        for base_name, models in model_groups.items():
            if len(models) > 1:
                models_with_stats = []
                for model in models:
                    stats = self.usage_stats.get(model, {'priority_score': 0})
                    models_with_stats.append((model, stats['priority_score']))
                
                models_with_stats.sort(key=lambda x: x[1], reverse=True)
                
                for model, score in models_with_stats[1:]:  # Pula o primeiro (mais prioritário)
                    duplicates.append({
                        'model': model,
                        'reason': f'Versão menos prioritária de {base_name}',
                        'priority_score': score
                    })
        
        return duplicates
    
    def _find_unused_models(self) -> List[str]:
        """Encontra modelos não utilizados"""
        unused = []
        
        for model_name in self.pretrained_loader.models_registry:
            if model_name not in self.usage_stats:
                unused.append(model_name)
            elif self._days_since_last_use(self.usage_stats[model_name]['last_used']) > 180:
                unused.append(model_name)
        
        return unused
    
    def _days_since_last_use(self, last_used_str: str) -> int:
        """Calcula dias desde último uso"""
        try:
            last_used = datetime.fromisoformat(last_used_str.replace('Z', '+00:00'))
            last_used = last_used.replace(tzinfo=None)
            return (datetime.now() - last_used).days
        except:
            return 999  # Valor alto se não conseguir calcular
    
    def _load_usage_stats(self) -> Dict[str, Any]:
        """Carrega estatísticas de uso do arquivo"""
        try:
            if self.usage_stats_file.exists():
                with open(self.usage_stats_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar estatísticas de uso: {e}")
        
        return {}
    
    def _save_usage_stats(self):
        """Salva estatísticas de uso no arquivo"""
        try:
            self.usage_stats_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.usage_stats_file, 'w') as f:
                json.dump(self.usage_stats, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Erro ao salvar estatísticas de uso: {e}")
    
    def _load_cache_policy(self) -> Dict[str, Any]:
        """Carrega política de cache do arquivo"""
        try:
            if self.cache_policy_file.exists():
                with open(self.cache_policy_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar política de cache: {e}")
        
        return {
            'auto_cleanup': True,
            'max_age_days': 90,
            'min_usage_threshold': 5,
            'priority_threshold': 0.5
        }
    
    def cleanup_unused_models(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Remove modelos não utilizados baseado na política de cache
        
        Args:
            dry_run: Se True, apenas simula a limpeza
            
        Returns:
            Relatório da limpeza
        """
        try:
            models_to_remove = self._identify_models_for_removal()
            
            if not models_to_remove:
                return {
                    'action': 'cleanup',
                    'models_removed': 0,
                    'space_freed_gb': 0,
                    'message': 'Nenhum modelo para remoção'
                }
            
            space_freed = 0
            removed_count = 0
            
            storage_info = self.pretrained_loader.get_storage_usage()
            
            for model_name in models_to_remove:
                if model_name in storage_info['model_sizes']:
                    model_size = storage_info['model_sizes'][model_name]
                    space_freed += model_size
                    
                    if not dry_run:
                        model_info = self.pretrained_loader.models_registry[model_name]
                        model_path = self.pretrained_loader.base_path.parent / model_info['file_path']
                        
                        if model_path.exists():
                            model_path.unlink()
                            removed_count += 1
                            logger.info(f"🗑️ Modelo removido: {model_name}")
            
            space_freed_gb = space_freed / (1024**3)
            
            return {
                'action': 'cleanup',
                'dry_run': dry_run,
                'models_removed': removed_count,
                'space_freed_gb': round(space_freed_gb, 2),
                'removed_models': models_to_remove[:removed_count] if not dry_run else models_to_remove,
                'message': f'{"Simulação: " if dry_run else ""}Removidos {removed_count} modelos, {space_freed_gb:.2f}GB liberados'
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na limpeza de modelos: {e}")
            return {
                'action': 'cleanup',
                'error': str(e),
                'models_removed': 0,
                'space_freed_gb': 0
            }
    
    def optimize_cache_automatically(self) -> Dict[str, Any]:
        """
        Executa otimização automática do cache baseada nas configurações
        
        Returns:
            Relatório da otimização
        """
        try:
            if not self.optimization_config['auto_cleanup_enabled']:
                return {'message': 'Otimização automática desabilitada'}
            
            optimization_results = {
                'timestamp': datetime.now().isoformat(),
                'actions_taken': [],
                'space_freed_gb': 0,
                'models_affected': 0
            }
            
            storage_info = self.pretrained_loader.get_storage_usage()
            current_size_gb = storage_info['total_size_gb']
            max_cache_gb = self.optimization_config['max_cache_size_gb']
            
            if current_size_gb > max_cache_gb:
                cleanup_result = self.cleanup_unused_models(dry_run=False)
                optimization_results['actions_taken'].append('cache_cleanup')
                optimization_results['space_freed_gb'] += cleanup_result['space_freed_gb']
                optimization_results['models_affected'] += cleanup_result['models_removed']
            
            corrupted_removed = self.pretrained_loader.cleanup_corrupted_models()
            if corrupted_removed > 0:
                optimization_results['actions_taken'].append('corrupted_cleanup')
                optimization_results['models_affected'] += corrupted_removed
            
            self._save_usage_stats()
            optimization_results['actions_taken'].append('stats_update')
            
            logger.info(f"✅ Otimização automática concluída: {len(optimization_results['actions_taken'])} ações")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"❌ Erro na otimização automática: {e}")
            return {'error': str(e)}
    
    def backup_usage_stats(self) -> bool:
        """Faz backup das estatísticas de uso"""
        try:
            backup_file = self.usage_stats_file.with_suffix(f'.backup_{int(time.time())}.json')
            
            if self.usage_stats_file.exists():
                import shutil
                shutil.copy2(self.usage_stats_file, backup_file)
                logger.info(f"✅ Backup criado: {backup_file}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro ao fazer backup: {e}")
            return False
