# dicom_processor.py - Processamento de arquivos DICOM

import pydicom
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import SimpleITK as sitk
from cryptography.fernet import Fernet
import json
import hashlib

logger = logging.getLogger('MedAI.DICOM')

class DICOMProcessor:
    """
    Classe para processamento de imagens DICOM médicas
    Suporta leitura, conversão, anonimização e pré-processamento
    """
    
    def __init__(self, anonymize: bool = True):
        self.anonymize = anonymize
        self._cache = {}
        self._fernet = Fernet(Fernet.generate_key()) if anonymize else None
        
    def read_dicom(self, filepath: Union[str, Path]) -> pydicom.Dataset:
        """
        Lê arquivo DICOM e retorna dataset
        
        Args:
            filepath: Caminho do arquivo DICOM
            
        Returns:
            Dataset DICOM
        """
        try:
            filepath = Path(filepath)
            
            # Verificar cache
            file_hash = self._get_file_hash(filepath)
            if file_hash in self._cache:
                logger.info(f"Arquivo {filepath.name} carregado do cache")
                return self._cache[file_hash]
            
            # Ler arquivo DICOM
            ds = pydicom.dcmread(str(filepath))
            
            # Anonimizar se necessário
            if self.anonymize:
                ds = self._anonymize_dicom(ds)
            
            # Adicionar ao cache
            self._cache[file_hash] = ds
            
            logger.info(f"DICOM carregado: {filepath.name}")
            return ds
            
        except Exception as e:
            logger.error(f"Erro ao ler DICOM {filepath}: {str(e)}")
            raise
    
    def _get_file_hash(self, filepath: Path) -> str:
        """Gera hash do arquivo para cache"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _anonymize_dicom(self, ds: pydicom.Dataset) -> pydicom.Dataset:
        """
        Anonimiza dados sensíveis do paciente
        
        Args:
            ds: Dataset DICOM
            
        Returns:
            Dataset anonimizado
        """
        # Tags para anonimizar
        sensitive_tags = [
            (0x0010, 0x0010),  # PatientName
            (0x0010, 0x0020),  # PatientID
            (0x0010, 0x0030),  # PatientBirthDate
            (0x0010, 0x1010),  # PatientAge
            (0x0010, 0x0040),  # PatientSex
            (0x0008, 0x0090),  # ReferringPhysicianName
            (0x0008, 0x1048),  # PhysiciansOfRecord
        ]
        
        # Criar cópia para não modificar original
        ds_anon = ds.copy()
        
        for tag in sensitive_tags:
            if tag in ds_anon:
                if tag == (0x0010, 0x0010):  # PatientName
                    ds_anon[tag].value = f"ANON_{hash(str(ds[tag].value)) % 10000:04d}"
                elif tag == (0x0010, 0x0020):  # PatientID
                    ds_anon[tag].value = f"ID_{hash(str(ds[tag].value)) % 100000:06d}"
                else:
                    ds_anon[tag].value = ""
        
        logger.info("DICOM anonimizado com sucesso")
        return ds_anon
    
    def dicom_to_array(self, ds: pydicom.Dataset) -> np.ndarray:
        """
        Converte pixel_array DICOM para array numpy normalizado
        
        Args:
            ds: Dataset DICOM
            
        Returns:
            Array numpy da imagem
        """
        # Obter array de pixels
        pixel_array = ds.pixel_array.astype(float)
        
        # Aplicar transformações DICOM (Rescale Slope/Intercept)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
        # Aplicar janelamento (Window Center/Width) se disponível
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            pixel_array = self._apply_windowing(
                pixel_array, 
                ds.WindowCenter, 
                ds.WindowWidth
            )
        
        # Normalizar para 0-255
        pixel_array = self._normalize_array(pixel_array)
        
        return pixel_array.astype(np.uint8)
    
    def _apply_windowing(self, 
                        image: np.ndarray, 
                        window_center: float, 
                        window_width: float) -> np.ndarray:
        """
        Aplica janelamento DICOM para melhor visualização
        
        Args:
            image: Array da imagem
            window_center: Centro da janela
            window_width: Largura da janela
            
        Returns:
            Imagem com janelamento aplicado
        """
        # Converter para valores únicos se forem listas
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = float(window_center[0])
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = float(window_width[0])
        
        # Calcular limites da janela
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        # Aplicar janelamento
        image = np.clip(image, window_min, window_max)
        
        return image
    
    def _normalize_array(self, array: np.ndarray) -> np.ndarray:
        """Normaliza array para range 0-255"""
        array_min = np.min(array)
        array_max = np.max(array)
        
        if array_max > array_min:
            array = ((array - array_min) / (array_max - array_min)) * 255
        else:
            array = np.zeros_like(array)
        
        return array
    
    def extract_metadata(self, ds: pydicom.Dataset) -> Dict:
        """
        Extrai metadados relevantes do DICOM
        
        Args:
            ds: Dataset DICOM
            
        Returns:
            Dicionário com metadados
        """
        metadata = {
            'StudyDate': str(ds.get('StudyDate', '')),
            'Modality': str(ds.get('Modality', '')),
            'BodyPartExamined': str(ds.get('BodyPartExamined', '')),
            'StudyDescription': str(ds.get('StudyDescription', '')),
            'SeriesDescription': str(ds.get('SeriesDescription', '')),
            'Rows': int(ds.get('Rows', 0)),
            'Columns': int(ds.get('Columns', 0)),
            'PixelSpacing': str(ds.get('PixelSpacing', '')),
            'SliceThickness': str(ds.get('SliceThickness', '')),
            'KVP': str(ds.get('KVP', '')),
            'ExposureTime': str(ds.get('ExposureTime', '')),
            'Manufacturer': str(ds.get('Manufacturer', '')),
            'ManufacturerModelName': str(ds.get('ManufacturerModelName', ''))
        }
        
        # Adicionar informações do paciente se não anonimizado
        if not self.anonymize:
            metadata.update({
                'PatientName': str(ds.get('PatientName', '')),
                'PatientID': str(ds.get('PatientID', '')),
                'PatientAge': str(ds.get('PatientAge', '')),
                'PatientSex': str(ds.get('PatientSex', ''))
            })
        
        return metadata
    
    def preprocess_for_ai(self, 
                         image: np.ndarray, 
                         target_size: Tuple[int, int],
                         normalize: bool = True) -> np.ndarray:
        """
        Pré-processa imagem para entrada em modelo de IA
        
        Args:
            image: Array da imagem
            target_size: Tamanho alvo (altura, largura)
            normalize: Se deve normalizar para [0, 1]
            
        Returns:
            Imagem pré-processada
        """
        # Redimensionar
        image = cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_LANCZOS4)
        
        # Equalização adaptativa de histograma
        if len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
        
        # Normalizar se necessário
        if normalize:
            image = image.astype(np.float32) / 255.0
        
        # Adicionar dimensão de canal se necessário
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        return image
    
    def save_as_png(self, image: np.ndarray, output_path: Union[str, Path]):
        """Salva imagem processada como PNG"""
        output_path = Path(output_path)
        cv2.imwrite(str(output_path), image)
        logger.info(f"Imagem salva: {output_path}")
