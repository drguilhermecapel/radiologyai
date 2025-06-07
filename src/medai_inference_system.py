# inference_system.py - Sistema de inferência e análise preditiva

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import time
from concurrent.futures import ThreadPoolExecutor
import queue

logger = logging.getLogger('MedAI.Inference')

@dataclass
class PredictionResult:
    """Estrutura para resultados de predição"""
    image_path: str
    predictions: Dict[str, float]
    predicted_class: str
    confidence: float
    processing_time: float
    heatmap: Optional[np.ndarray] = None
    attention_map: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    
class MedicalInferenceEngine:
    """
    Motor de inferência otimizado para análise de imagens médicas
    Suporta batch processing, visualização de atenção e análise em tempo real
    """
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 model_config: Dict,
                 use_gpu: bool = True,
                 batch_size: int = 32):
        self.model_path = Path(model_path)
        self.model_config = model_config
        self.batch_size = batch_size
        self.model = None
        self.preprocessing_fn = None
        
        # Configurar GPU/CPU
        self._configure_device(use_gpu)
        
        # Carregar modelo
        self._load_model()
        
        # Cache para otimização
        self._prediction_cache = {}
        self._max_cache_size = 1000
        
    def _configure_device(self, use_gpu: bool):
        """Configura dispositivo de processamento"""
        if use_gpu and tf.config.list_physical_devices('GPU'):
            logger.info("GPU detectada, usando aceleração por GPU")
            # Configurar crescimento de memória GPU
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            logger.info("Usando CPU para processamento")
            tf.config.set_visible_devices([], 'GPU')
    
    def _load_model(self):
        """Carrega modelo treinado"""
        try:
            self.model = tf.keras.models.load_model(
                self.model_path,
                compile=False  # Compilar manualmente para controle
            )
            
            # Recompilar com métricas de inferência
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"Modelo carregado: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise
    
    def predict_single(self, 
                      image: np.ndarray,
                      return_attention: bool = True,
                      metadata: Optional[Dict] = None) -> PredictionResult:
        """
        Realiza predição em uma única imagem
        
        Args:
            image: Array da imagem
            return_attention: Se deve retornar mapas de atenção
            metadata: Metadados da imagem
            
        Returns:
            Resultado da predição
        """
        start_time = time.time()
        
        # Verificar cache
        image_hash = hash(image.tobytes())
        if image_hash in self._prediction_cache:
            logger.info("Predição encontrada no cache")
            return self._prediction_cache[image_hash]
        
        # Pré-processar imagem
        processed_image = self._preprocess_image(image)
        
        # Expandir dimensões para batch
        batch_image = np.expand_dims(processed_image, axis=0)
        
        # Predição
        predictions = self.model.predict(batch_image, verbose=0)[0]
        
        # Processar resultados
        class_names = self.model_config['classes']
        prediction_dict = {
            class_names[i]: float(predictions[i]) 
            for i in range(len(class_names))
        }
        
        # Classe predita
        predicted_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Gerar mapas de atenção se solicitado
        heatmap = None
        attention_map = None
        if return_attention:
            heatmap = self._generate_gradcam_heatmap(batch_image, predicted_idx)
            attention_map = self._generate_attention_map(batch_image)
        
        # Criar resultado
        result = PredictionResult(
            image_path="",
            predictions=prediction_dict,
            predicted_class=predicted_class,
            confidence=confidence,
            processing_time=time.time() - start_time,
            heatmap=heatmap,
            attention_map=attention_map,
            metadata=metadata
        )
        
        # Adicionar ao cache
        self._add_to_cache(image_hash, result)
        
        return result
    
    def predict_batch(self,
                     images: List[np.ndarray],
                     return_attention: bool = False,
                     use_multiprocessing: bool = True) -> List[PredictionResult]:
        """
        Realiza predição em lote de imagens
        
        Args:
            images: Lista de arrays de imagens
            return_attention: Se deve retornar mapas de atenção
            use_multiprocessing: Se deve usar processamento paralelo
            
        Returns:
            Lista de resultados
        """
        logger.info(f"Processando lote de {len(images)} imagens")
        
        if use_multiprocessing and len(images) > 10:
            # Processamento paralelo para lotes grandes
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(
                    lambda img: self.predict_single(img, return_attention),
                    images
                ))
        else:
            # Processamento sequencial
            results = []
            
            # Processar em mini-lotes
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size]
                
                # Pré-processar lote
                processed_batch = np.array([
                    self._preprocess_image(img) for img in batch
                ])
                
                # Predição em lote
                batch_predictions = self.model.predict(
                    processed_batch, 
                    verbose=0
                )
                
                # Processar resultados
                for j, (img, preds) in enumerate(zip(batch, batch_predictions)):
                    class_names = self.model_config['classes']
                    prediction_dict = {
                        class_names[k]: float(preds[k]) 
                        for k in range(len(class_names))
                    }
                    
                    predicted_idx = np.argmax(preds)
                    
                    result = PredictionResult(
                        image_path=f"batch_{i+j}",
                        predictions=prediction_dict,
                        predicted_class=class_names[predicted_idx],
                        confidence=float(preds[predicted_idx]),
                        processing_time=0.0
                    )
                    
                    results.append(result)
        
        return results
    
    def stream_predict(self, 
                      image_queue: queue.Queue,
                      result_queue: queue.Queue,
                      stop_event):
        """
        Predição em streaming para análise em tempo real
        
        Args:
            image_queue: Fila de entrada de imagens
            result_queue: Fila de saída de resultados
            stop_event: Evento para parar o streaming
        """
        logger.info("Iniciando predição em streaming")
        
        while not stop_event.is_set():
            try:
                # Obter imagem da fila (timeout para verificar stop_event)
                image = image_queue.get(timeout=0.1)
                
                # Realizar predição
                result = self.predict_single(image, return_attention=False)
                
                # Adicionar resultado à fila
                result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Erro no streaming: {str(e)}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Pré-processa imagem para o modelo
        
        Args:
            image: Imagem original
            
        Returns:
            Imagem pré-processada
        """
        target_size = self.model_config['input_size']
        
        # Redimensionar
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size[::-1], 
                             interpolation=cv2.INTER_LANCZOS4)
        
        # Normalizar
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        if np.max(image) > 1.0:
            image = image / 255.0
        
        # Aplicar CLAHE se for escala de cinza
        if len(image.shape) == 2 or image.shape[-1] == 1:
            image_uint8 = (image * 255).astype(np.uint8)
            if len(image.shape) == 3:
                image_uint8 = image_uint8[:, :, 0]
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image_uint8 = clahe.apply(image_uint8)
            
            image = image_uint8.astype(np.float32) / 255.0
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
        
        # Garantir 3 canais se necessário
        if self.model.input_shape[-1] == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        return image
    
    def _generate_gradcam_heatmap(self, 
                                 image: np.ndarray,
                                 class_idx: int) -> np.ndarray:
        """
        Gera mapa de calor Grad-CAM para visualização de atenção
        
        Args:
            image: Imagem de entrada
            class_idx: Índice da classe predita
            
        Returns:
            Mapa de calor
        """
        # Encontrar última camada convolucional
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, 
                                tf.keras.layers.Conv2DTranspose)):
                last_conv_layer = layer
                break
        
        if not last_conv_layer:
            return None
        
        # Criar modelo para obter saídas intermediárias
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [last_conv_layer.output, self.model.output]
        )
        
        # Computar gradientes
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, class_idx]
        
        # Gradientes da classe em relação à saída da última conv
        grads = tape.gradient(loss, conv_outputs)
        
        # Pooling dos gradientes
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Ponderar canais pelos gradientes
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalizar heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Redimensionar para tamanho original
        heatmap = cv2.resize(heatmap, self.model_config['input_size'][::-1])
        
        return heatmap
    
    def _generate_attention_map(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Gera mapa de atenção se o modelo tiver camadas de atenção
        
        Args:
            image: Imagem de entrada
            
        Returns:
            Mapa de atenção ou None
        """
        # Procurar por camadas de atenção
        attention_outputs = []
        
        for layer in self.model.layers:
            if 'attention' in layer.name.lower():
                # Criar modelo intermediário
                attention_model = tf.keras.models.Model(
                    inputs=self.model.input,
                    outputs=layer.output
                )
                
                # Obter saída da camada
                attention_output = attention_model.predict(image, verbose=0)
                attention_outputs.append(attention_output)
        
        if attention_outputs:
            # Combinar mapas de atenção
            combined_attention = np.mean(attention_outputs, axis=0)
            
            # Processar para visualização
            if len(combined_attention.shape) > 2:
                combined_attention = np.mean(combined_attention, axis=-1)
            
            # Normalizar
            combined_attention = (combined_attention - np.min(combined_attention)) / \
                               (np.max(combined_attention) - np.min(combined_attention))
            
            return combined_attention[0]
        
        return None
    
    def visualize_prediction(self, 
                           image: np.ndarray,
                           result: PredictionResult,
                           save_path: Optional[Path] = None) -> plt.Figure:
        """
        Visualiza resultado da predição com mapas de atenção
        
        Args:
            image: Imagem original
            result: Resultado da predição
            save_path: Caminho para salvar visualização
            
        Returns:
            Figura matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Imagem original
        axes[0, 0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        axes[0, 0].set_title('Imagem Original')
        axes[0, 0].axis('off')
        
        # Predições
        classes = list(result.predictions.keys())
        probs = list(result.predictions.values())
        
        axes[0, 1].barh(classes, probs)
        axes[0, 1].set_xlabel('Probabilidade')
        axes[0, 1].set_title('Predições')
        axes[0, 1].set_xlim(0, 1)
        
        # Adicionar valores nas barras
        for i, (cls, prob) in enumerate(zip(classes, probs)):
            axes[0, 1].text(prob + 0.01, i, f'{prob:.3f}', 
                          va='center', fontsize=10)
        
        # Heatmap Grad-CAM
        if result.heatmap is not None:
            # Sobrepor heatmap na imagem
            heatmap_colored = plt.cm.jet(result.heatmap)[:, :, :3]
            
            if len(image.shape) == 2:
                image_rgb = np.stack([image] * 3, axis=-1)
            else:
                image_rgb = image
            
            # Normalizar imagem para sobreposição
            if np.max(image_rgb) > 1:
                image_rgb = image_rgb / 255.0
            
            superimposed = 0.6 * image_rgb + 0.4 * heatmap_colored
            
            axes[1, 0].imshow(superimposed)
            axes[1, 0].set_title('Mapa de Calor Grad-CAM')
            axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'Grad-CAM não disponível', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
        
        # Informações da predição
        info_text = f"""
        Classe Predita: {result.predicted_class}
        Confiança: {result.confidence:.2%}
        Tempo de Processamento: {result.processing_time:.3f}s
        
        Limiar de Decisão: {self.model_config.get('threshold', 0.5)}
        """
        
        axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Análise de Predição - {result.predicted_class}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização salva em: {save_path}")
        
        return fig
    
    def _add_to_cache(self, key: int, result: PredictionResult):
        """Adiciona resultado ao cache com limite de tamanho"""
        if len(self._prediction_cache) >= self._max_cache_size:
            # Remover entrada mais antiga
            oldest_key = next(iter(self._prediction_cache))
            del self._prediction_cache[oldest_key]
        
        self._prediction_cache[key] = result
    
    def analyze_uncertainty(self, 
                          predictions: np.ndarray,
                          n_classes: int) -> Dict[str, float]:
        """
        Analisa incerteza das predições usando entropia e outras métricas
        
        Args:
            predictions: Array de probabilidades
            n_classes: Número de classes
            
        Returns:
            Métricas de incerteza
        """
        # Entropia
        entropy = -np.sum(predictions * np.log(predictions + 1e-10))
        max_entropy = np.log(n_classes)
        normalized_entropy = entropy / max_entropy
        
        # Diferença entre top-2 predições
        sorted_preds = np.sort(predictions)[::-1]
        margin = sorted_preds[0] - sorted_preds[1] if len(sorted_preds) > 1 else 1.0
        
        # Variância das predições
        variance = np.var(predictions)
        
        return {
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'margin': float(margin),
            'variance': float(variance),
            'max_probability': float(np.max(predictions))
        }
