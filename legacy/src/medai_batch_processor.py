# batch_processor.py - Sistema de processamento em lote para análise de múltiplas imagens

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
import logging
import time
from tqdm import tqdm
import json
import csv
from datetime import datetime
import psutil
import os
from queue import Queue, Empty
import threading

logger = logging.getLogger('MedAI.BatchProcessor')

@dataclass
class BatchJob:
    """Estrutura para representar um trabalho em lote"""
    job_id: str
    input_files: List[Path]
    output_dir: Path
    model_type: str
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, processing, completed, failed
    progress: float = 0.0
    results: List[Dict] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    
class BatchProcessor:
    """
    Processador em lote otimizado para análise de grandes volumes de imagens médicas
    Suporta processamento paralelo, monitoramento de progresso e recuperação de falhas
    """
    
    def __init__(self,
                 inference_engine,
                 max_workers: Optional[int] = None,
                 use_gpu: bool = True,
                 batch_size: int = 32,
                 memory_limit_gb: float = 8.0):
        
        self.inference_engine = inference_engine
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.memory_limit_gb = memory_limit_gb
        
        # Determinar número de workers
        if max_workers is None:
            # Se GPU, usar menos workers para evitar conflitos
            if use_gpu:
                max_workers = 2
            else:
                max_workers = multiprocessing.cpu_count() - 1
        
        self.max_workers = max_workers
        
        # Filas para processamento
        self.job_queue = Queue()
        self.result_queue = Queue()
        
        # Estado do processador
        self.active_jobs: Dict[str, BatchJob] = {}
        self.is_running = False
        self._lock = threading.Lock()
        
        # Estatísticas
        self.stats = {
            'total_processed': 0,
            'total_failed': 0,
            'total_time': 0.0,
            'average_time_per_image': 0.0
        }
        
        logger.info(f"BatchProcessor inicializado com {self.max_workers} workers")
    
    def create_batch_job(self,
                        input_paths: List[str],
                        output_dir: str,
                        model_type: str,
                        recursive: bool = False) -> BatchJob:
        """
        Cria um novo trabalho em lote
        
        Args:
            input_paths: Lista de caminhos de entrada (arquivos ou diretórios)
            output_dir: Diretório de saída
            model_type: Tipo de modelo a usar
            recursive: Se deve buscar arquivos recursivamente em diretórios
            
        Returns:
            Objeto BatchJob criado
        """
        # Coletar todos os arquivos
        all_files = []
        
        for path_str in input_paths:
            path = Path(path_str)
            
            if path.is_file():
                all_files.append(path)
            elif path.is_dir():
                # Buscar arquivos de imagem no diretório
                patterns = ['*.dcm', '*.dicom', '*.png', '*.jpg', '*.jpeg', 
                           '*.nii', '*.nii.gz']
                
                for pattern in patterns:
                    if recursive:
                        all_files.extend(path.rglob(pattern))
                    else:
                        all_files.extend(path.glob(pattern))
        
        # Remover duplicatas
        all_files = list(set(all_files))
        
        # Criar job
        job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(all_files)}"
        
        job = BatchJob(
            job_id=job_id,
            input_files=all_files,
            output_dir=Path(output_dir),
            model_type=model_type
        )
        
        # Criar diretório de saída
        job.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Registrar job
        with self._lock:
            self.active_jobs[job_id] = job
        
        logger.info(f"Job criado: {job_id} com {len(all_files)} arquivos")
        
        return job
    
    def process_batch(self, 
                     job: BatchJob,
                     progress_callback: Optional[Callable] = None) -> Dict:
        """
        Processa um trabalho em lote
        
        Args:
            job: Trabalho a processar
            progress_callback: Função callback para progresso
            
        Returns:
            Dicionário com resultados do processamento
        """
        start_time = time.time()
        job.status = "processing"
        
        logger.info(f"Iniciando processamento do job {job.job_id}")
        
        try:
            # Dividir arquivos em chunks para processamento paralelo
            chunks = self._create_chunks(job.input_files, self.batch_size)
            total_chunks = len(chunks)
            
            # Processar chunks em paralelo
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submeter todos os chunks
                futures = []
                for i, chunk in enumerate(chunks):
                    future = executor.submit(
                        self._process_chunk,
                        chunk,
                        job,
                        i,
                        total_chunks
                    )
                    futures.append(future)
                
                # Coletar resultados
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        chunk_results = future.result()
                        job.results.extend(chunk_results['successes'])
                        job.errors.extend(chunk_results['failures'])
                        
                        completed += 1
                        job.progress = (completed / total_chunks) * 100
                        
                        if progress_callback:
                            progress_callback(job)
                        
                    except Exception as e:
                        logger.error(f"Erro ao processar chunk: {str(e)}")
                        job.errors.append({
                            'error': str(e),
                            'type': 'chunk_processing'
                        })
            
            # Finalizar job
            job.status = "completed"
            job.progress = 100.0
            
            # Salvar resultados
            self._save_results(job)
            
            # Atualizar estatísticas
            elapsed_time = time.time() - start_time
            self._update_stats(job, elapsed_time)
            
            # Gerar relatório resumido
            summary = self._generate_summary(job, elapsed_time)
            
            logger.info(f"Job {job.job_id} concluído em {elapsed_time:.2f}s")
            
            return summary
            
        except Exception as e:
            job.status = "failed"
            job.errors.append({
                'error': str(e),
                'type': 'job_processing'
            })
            logger.error(f"Falha no job {job.job_id}: {str(e)}")
            raise
    
    def _create_chunks(self, files: List[Path], chunk_size: int) -> List[List[Path]]:
        """Divide lista de arquivos em chunks para processamento paralelo"""
        chunks = []
        for i in range(0, len(files), chunk_size):
            chunks.append(files[i:i + chunk_size])
        return chunks
    
    def _process_chunk(self,
                      files: List[Path],
                      job: BatchJob,
                      chunk_index: int,
                      total_chunks: int) -> Dict:
        """Processa um chunk de arquivos"""
        chunk_results = {
            'successes': [],
            'failures': []
        }
        
        # Verificar memória disponível
        if not self._check_memory():
            logger.warning(f"Memória insuficiente para chunk {chunk_index}")
            # Aguardar liberação de memória
            time.sleep(5)
        
        for file_path in files:
            try:
                # Processar arquivo individual
                result = self._process_single_file(file_path, job.model_type)
                
                # Adicionar caminho ao resultado
                result['file_path'] = str(file_path)
                result['file_name'] = file_path.name
                
                chunk_results['successes'].append(result)
                
            except Exception as e:
                logger.error(f"Erro ao processar {file_path}: {str(e)}")
                chunk_results['failures'].append({
                    'file_path': str(file_path),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return chunk_results
    
    def _process_single_file(self, file_path: Path, model_type: str) -> Dict:
        """Processa um único arquivo"""
        # Carregar imagem
        if file_path.suffix.lower() in ['.dcm', '.dicom']:
            # Processar DICOM
            from dicom_processor import DICOMProcessor
            processor = DICOMProcessor()
            ds = processor.read_dicom(file_path)
            image = processor.dicom_to_array(ds)
            metadata = processor.extract_metadata(ds)
        else:
            # Processar imagem comum
            import cv2
            image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            metadata = {'file_type': file_path.suffix}
        
        # Realizar inferência
        result = self.inference_engine.predict_single(
            image,
            return_attention=False,  # Desabilitar para batch
            metadata=metadata
        )
        
        # Converter resultado para dicionário serializável
        return {
            'predictions': result.predictions,
            'predicted_class': result.predicted_class,
            'confidence': result.confidence,
            'processing_time': result.processing_time,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_memory(self) -> bool:
        """Verifica se há memória suficiente disponível"""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        return available_gb > self.memory_limit_gb * 0.2  # 20% do limite
    
    def _save_results(self, job: BatchJob):
        """Salva resultados do processamento"""
        # Salvar resultados em CSV
        csv_path = job.output_dir / f"{job.job_id}_results.csv"
        
        if job.results:
            # Preparar dados para CSV
            csv_data = []
            for result in job.results:
                row = {
                    'file_path': result['file_path'],
                    'file_name': result['file_name'],
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence'],
                    'processing_time': result['processing_time'],
                    'timestamp': result['timestamp']
                }
                
                # Adicionar probabilidades individuais
                for class_name, prob in result['predictions'].items():
                    row[f'prob_{class_name}'] = prob
                
                csv_data.append(row)
            
            # Criar DataFrame e salvar
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            logger.info(f"Resultados salvos em: {csv_path}")
        
        # Salvar erros se houver
        if job.errors:
            errors_path = job.output_dir / f"{job.job_id}_errors.json"
            with open(errors_path, 'w') as f:
                json.dump(job.errors, f, indent=4)
            logger.info(f"Erros salvos em: {errors_path}")
        
        # Salvar relatório JSON completo
        report_path = job.output_dir / f"{job.job_id}_report.json"
        report = {
            'job_id': job.job_id,
            'model_type': job.model_type,
            'created_at': job.created_at.isoformat(),
            'total_files': len(job.input_files),
            'processed_files': len(job.results),
            'failed_files': len(job.errors),
            'success_rate': len(job.results) / len(job.input_files) if job.input_files else 0,
            'output_dir': str(job.output_dir)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
    
    def _update_stats(self, job: BatchJob, elapsed_time: float):
        """Atualiza estatísticas globais"""
        with self._lock:
            self.stats['total_processed'] += len(job.results)
            self.stats['total_failed'] += len(job.errors)
            self.stats['total_time'] += elapsed_time
            
            if self.stats['total_processed'] > 0:
                self.stats['average_time_per_image'] = \
                    self.stats['total_time'] / self.stats['total_processed']
    
    def _generate_summary(self, job: BatchJob, elapsed_time: float) -> Dict:
        """Gera resumo do processamento"""
        summary = {
            'job_id': job.job_id,
            'status': job.status,
            'total_files': len(job.input_files),
            'processed': len(job.results),
            'failed': len(job.errors),
            'success_rate': f"{(len(job.results) / len(job.input_files) * 100):.1f}%",
            'total_time': f"{elapsed_time:.2f}s",
            'average_time_per_file': f"{elapsed_time / len(job.input_files):.2f}s",
            'output_directory': str(job.output_dir)
        }
        
        # Análise de resultados
        if job.results:
            # Contar classes preditas
            class_counts = {}
            confidence_sum = 0
            
            for result in job.results:
                pred_class = result['predicted_class']
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                confidence_sum += result['confidence']
            
            summary['class_distribution'] = class_counts
            summary['average_confidence'] = f"{confidence_sum / len(job.results):.2%}"
        
        return summary
    
    def process_directory_watch(self,
                              watch_dir: str,
                              output_dir: str,
                              model_type: str,
                              interval: int = 10):
        """
        Monitora diretório e processa novos arquivos automaticamente
        
        Args:
            watch_dir: Diretório a monitorar
            output_dir: Diretório de saída
            model_type: Tipo de modelo
            interval: Intervalo de verificação em segundos
        """
        watch_path = Path(watch_dir)
        processed_files = set()
        
        logger.info(f"Monitorando diretório: {watch_path}")
        
        try:
            while True:
                # Buscar novos arquivos
                current_files = set()
                for pattern in ['*.dcm', '*.png', '*.jpg']:
                    current_files.update(watch_path.glob(pattern))
                
                # Identificar novos arquivos
                new_files = current_files - processed_files
                
                if new_files:
                    logger.info(f"Encontrados {len(new_files)} novos arquivos")
                    
                    # Criar job para novos arquivos
                    job = self.create_batch_job(
                        [str(f) for f in new_files],
                        output_dir,
                        model_type
                    )
                    
                    # Processar
                    self.process_batch(job)
                    
                    # Marcar como processados
                    processed_files.update(new_files)
                
                # Aguardar próxima verificação
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoramento interrompido")
    
    def generate_batch_report(self, job_id: str) -> str:
        """
        Gera relatório detalhado de um job
        
        Args:
            job_id: ID do job
            
        Returns:
            Caminho do relatório gerado
        """
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} não encontrado")
        
        job = self.active_jobs[job_id]
        
        # Criar relatório HTML
        report_path = job.output_dir / f"{job_id}_detailed_report.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Relatório de Processamento em Lote - {job_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #003366; color: white; padding: 20px; }}
        .summary {{ background-color: #f0f0f0; padding: 15px; margin: 20px 0; }}
        .results-table {{ width: 100%; border-collapse: collapse; }}
        .results-table th, .results-table td {{ 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: left; 
        }}
        .results-table th {{ background-color: #0066cc; color: white; }}
        .results-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .confidence-high {{ color: green; font-weight: bold; }}
        .confidence-medium {{ color: orange; font-weight: bold; }}
        .confidence-low {{ color: red; font-weight: bold; }}
        .error {{ background-color: #ffcccc; }}
        .chart {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Relatório de Processamento em Lote</h1>
        <p>Job ID: {job_id}</p>
        <p>Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
    </div>
    
    <div class="summary">
        <h2>Resumo</h2>
        <p><strong>Total de Arquivos:</strong> {len(job.input_files)}</p>
        <p><strong>Processados com Sucesso:</strong> {len(job.results)}</p>
        <p><strong>Falhas:</strong> {len(job.errors)}</p>
        <p><strong>Taxa de Sucesso:</strong> {len(job.results) / len(job.input_files) * 100:.1f}%</p>
        <p><strong>Modelo Utilizado:</strong> {job.model_type}</p>
    </div>
    
    <h2>Resultados Detalhados</h2>
    <table class="results-table">
        <thead>
            <tr>
                <th>Arquivo</th>
                <th>Classe Predita</th>
                <th>Confiança</th>
                <th>Tempo (s)</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
"""
        
        # Adicionar resultados bem-sucedidos
        for result in sorted(job.results, key=lambda x: x['file_name']):
            confidence = result['confidence']
            conf_class = 'confidence-high' if confidence > 0.8 else \
                        'confidence-medium' if confidence > 0.6 else \
                        'confidence-low'
            
            html_content += f"""
            <tr>
                <td>{result['file_name']}</td>
                <td>{result['predicted_class']}</td>
                <td class="{conf_class}">{confidence:.1%}</td>
                <td>{result['processing_time']:.3f}</td>
                <td>✓ Sucesso</td>
            </tr>
"""
        
        # Adicionar erros
        for error in job.errors:
            html_content += f"""
            <tr class="error">
                <td>{Path(error['file_path']).name}</td>
                <td colspan="3">{error['error']}</td>
                <td>✗ Erro</td>
            </tr>
"""
        
        html_content += """
        </tbody>
    </table>
    
    <div class="chart">
        <h2>Distribuição de Classes</h2>
        <canvas id="classChart" width="400" height="200"></canvas>
    </div>
    
    <script>
        // Adicionar gráfico de distribuição se necessário
    </script>
</body>
</html>
"""
        
        # Salvar relatório
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Relatório detalhado gerado: {report_path}")
        return str(report_path)
