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

try:
    from .medai_modality_normalizer import ModalitySpecificNormalizer
except ImportError:
    try:
        from medai_modality_normalizer import ModalitySpecificNormalizer
    except ImportError:
        ModalitySpecificNormalizer = None

logger = logging.getLogger('MedAI.DICOM')

class DicomProcessor:
    """
    Classe para processamento de imagens DICOM médicas
    Suporta leitura, conversão, anonimização e pré-processamento
    Inclui suporte para modalidades: CR, CT, MR, US, MG, DX
    """
    
    SUPPORTED_MODALITIES = {
        'CR': 'Computed Radiography',
        'CT': 'Computed Tomography', 
        'MR': 'Magnetic Resonance',
        'US': 'Ultrasound',
        'MG': 'Mammography',
        'DX': 'Digital Radiography',
        'XA': 'X-Ray Angiography',
        'RF': 'Radio Fluoroscopy',
        'PT': 'Positron Emission Tomography'
    }
    
    def __init__(self, anonymize: bool = True):
        self.anonymize = anonymize
        self._cache = {}
        self._fernet = Fernet(Fernet.generate_key()) if anonymize else None
        
        if ModalitySpecificNormalizer:
            self.normalizer = ModalitySpecificNormalizer()
        else:
            self.normalizer = None
        
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
            
            # Verificar se arquivo existe
            if not filepath.exists():
                logger.error(f"Arquivo DICOM não encontrado: {filepath}")
                raise FileNotFoundError(f"Arquivo DICOM não encontrado: {filepath}")
            
            # Verificar cache
            file_hash = self._get_file_hash(filepath)
            if file_hash in self._cache:
                logger.info(f"Arquivo {filepath.name} carregado do cache")
                return self._cache[file_hash]
            
            try:
                ds = pydicom.dcmread(str(filepath))
            except Exception as e:
                if "DICM" in str(e) or "File Meta Information" in str(e):
                    logger.warning(f"Tentando ler DICOM com force=True: {filepath}")
                    ds = pydicom.dcmread(str(filepath), force=True)
                else:
                    raise e
            
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
        Converte pixel_array DICOM para array numpy com normalização específica por modalidade
        Implementa conversão adequada de Hounsfield Units para CT e outras modalidades
        
        Args:
            ds: Dataset DICOM
            
        Returns:
            Array numpy da imagem normalizada
        """
        try:
            # Usar normalização específica por modalidade se disponível
            if self.normalizer is not None:
                modality = str(ds.get('Modality', 'CR'))
                
                if modality == 'CT':
                    normalized_array = self.normalizer.normalize_ct(ds, target_organ='soft_tissue')
                    # Converter para uint8 para compatibilidade
                    return (normalized_array * 255).astype(np.uint8)
                    
                elif modality in ['MR', 'MRI']:
                    pixel_array = ds.pixel_array.astype(float)
                    normalized_array = self.normalizer.normalize_mri(pixel_array, sequence_type='T1')
                    return (normalized_array * 255).astype(np.uint8)
                    
                elif modality in ['CR', 'DX']:
                    pixel_array = ds.pixel_array.astype(float)
                    normalized_array = self.normalizer.normalize_xray(pixel_array, enhance_contrast=True)
                    return (normalized_array * 255).astype(np.uint8)
                    
                else:
                    pixel_array = ds.pixel_array.astype(float)
                    normalized_array = self.normalizer.normalize_by_modality(pixel_array, modality)
                    return (normalized_array * 255).astype(np.uint8)
            
            # Fallback para método tradicional se normalizer não disponível
            logger.warning("ModalitySpecificNormalizer não disponível, usando método tradicional")
            return self._legacy_dicom_to_array(ds)
            
        except Exception as e:
            logger.error(f"Erro na conversão DICOM com normalização específica: {e}")
            return self._legacy_dicom_to_array(ds)
    
    def _legacy_dicom_to_array(self, ds: pydicom.Dataset) -> np.ndarray:
        """Método tradicional de conversão DICOM como fallback"""
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
        pixel_array = self._normalize_array_legacy(pixel_array)
        
        return pixel_array.astype(np.uint8)
    
    def _apply_windowing(self, 
                        image: np.ndarray, 
                        window_center: Union[float, pydicom.multival.MultiValue], 
                        window_width: Union[float, pydicom.multival.MultiValue]) -> np.ndarray:
        """
        Aplica janelamento DICOM para melhor visualização
        Corrige problemas de tipo com MultiValue
        
        Args:
            image: Array da imagem
            window_center: Centro da janela
            window_width: Largura da janela
            
        Returns:
            Imagem com janelamento aplicado
        """
        try:
            # Converter para valores únicos se forem MultiValue do pydicom
            try:
                window_center = float(window_center)
            except (TypeError, ValueError):
                try:
                    window_center = float(window_center[0])
                except (TypeError, IndexError):
                    window_center = 0.0  # Valor padrão seguro
                
            try:
                window_width = float(window_width)
            except (TypeError, ValueError):
                try:
                    window_width = float(window_width[0])
                except (TypeError, IndexError):
                    window_width = 1.0  # Valor padrão seguro
            
            # Calcular limites da janela
            window_min = window_center - window_width / 2
            window_max = window_center + window_width / 2
            
            # Aplicar janelamento
            image = np.clip(image, window_min, window_max)
            
            logger.debug(f"Janelamento aplicado: WC={window_center}, WW={window_width}")
            return image
            
        except Exception as e:
            logger.warning(f"Erro no janelamento DICOM: {e}. Retornando imagem original.")
            return image
    
    def _normalize_array_legacy(self, array: np.ndarray) -> np.ndarray:
        """
        Normalização legacy para range 0-255 (usado como fallback)
        Substituído pela normalização específica por modalidade
        """
        try:
            p1, p99 = np.percentile(array, [1, 99])
            
            if p99 > p1:
                array = np.clip(array, p1, p99)
                array = ((array - p1) / (p99 - p1)) * 255
            else:
                array = np.zeros_like(array)
            
            return array
        except:
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
                         modality: str = 'CR',
                         normalize: bool = True,
                         ds: Optional[pydicom.Dataset] = None) -> np.ndarray:
        """
        Pré-processa imagem para entrada em modelo de IA
        Usa normalização específica por modalidade quando disponível
        
        Args:
            image: Array da imagem
            target_size: Tamanho alvo (altura, largura)
            modality: Modalidade médica (CR, CT, MR, US, MG, etc.)
            normalize: Se deve normalizar para [0, 1]
            ds: Dataset DICOM opcional para normalização avançada
            
        Returns:
            Imagem pré-processada com normalização médica otimizada
        """
        try:
            if ds is not None and self.normalizer is not None:
                if modality == 'CT':
                    normalized_image = self.normalizer.normalize_ct(ds, target_organ='soft_tissue')
                elif modality in ['MR', 'MRI']:
                    normalized_image = self.normalizer.normalize_mri(image, sequence_type='T1')
                elif modality in ['CR', 'DX']:
                    normalized_image = self.normalizer.normalize_xray(image, enhance_contrast=True)
                else:
                    normalized_image = self.normalizer.normalize_by_modality(image, modality)
                
                # Redimensionar imagem normalizada
                if normalized_image.dtype != np.uint8:
                    normalized_image = (normalized_image * 255).astype(np.uint8)
                
                image = cv2.resize(normalized_image, target_size[::-1], interpolation=cv2.INTER_LANCZOS4)
                
                # Converter de volta para float se necessário
                if normalize:
                    image = image.astype(np.float32) / 255.0
                    
            else:
                image = cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_LANCZOS4)
                image = self._apply_modality_preprocessing(image, modality)
                
                if normalize:
                    image = image.astype(np.float32) / 255.0
            
            # Adicionar dimensão de canal se necessário
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            
            logger.info(f"Pré-processamento AI aplicado para modalidade {modality}")
            return image
            
        except Exception as e:
            logger.error(f"Erro no pré-processamento AI: {e}. Usando método tradicional.")
            image = cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_LANCZOS4)
            image = self._apply_modality_preprocessing(image, modality)
            
            if normalize:
                image = image.astype(np.float32) / 255.0
            
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            
            return image
    
    def _apply_modality_preprocessing(self, image: np.ndarray, modality: str) -> np.ndarray:
        """
        Aplica pré-processamento específico por modalidade
        
        Args:
            image: Array da imagem
            modality: Modalidade médica
            
        Returns:
            Imagem pré-processada
        """
        if len(image.shape) == 2:
            if modality in ['CR', 'DX']:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image = clahe.apply(image)
                
            elif modality == 'CT':
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
                image = clahe.apply(image)
                
            elif modality == 'MR':
                clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(12, 12))
                image = clahe.apply(image)
                
            elif modality == 'US':
                image = cv2.medianBlur(image, 3)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
                image = clahe.apply(image)
                
            elif modality == 'MG':
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
                image = clahe.apply(image)
                
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image = clahe.apply(image)
        
        return image
    
    def save_as_png(self, image: np.ndarray, output_path: Union[str, Path]):
        """Salva imagem processada como PNG"""
        output_path = Path(output_path)
        cv2.imwrite(str(output_path), image)
        logger.info(f"Imagem salva: {output_path}")
    
    def get_modality_info(self, ds: pydicom.Dataset) -> Dict[str, str]:
        """
        Obtém informações específicas da modalidade
        
        Args:
            ds: Dataset DICOM
            
        Returns:
            Dicionário com informações da modalidade
        """
        modality = str(ds.get('Modality', 'Unknown'))
        
        modality_info = {
            'modality': modality,
            'description': self.SUPPORTED_MODALITIES.get(modality, 'Unknown Modality'),
            'supported': modality in self.SUPPORTED_MODALITIES
        }
        
        if modality == 'US':
            modality_info.update({
                'transducer_frequency': str(ds.get('TransducerFrequency', '')),
                'ultrasound_color_data_present': str(ds.get('UltrasoundColorDataPresent', ''))
            })
        elif modality == 'MG':
            modality_info.update({
                'view_position': str(ds.get('ViewPosition', '')),
                'compression_force': str(ds.get('CompressionForce', '')),
                'breast_implant_present': str(ds.get('BreastImplantPresent', ''))
            })
        elif modality == 'CT':
            modality_info.update({
                'slice_thickness': str(ds.get('SliceThickness', '')),
                'reconstruction_diameter': str(ds.get('ReconstructionDiameter', '')),
                'convolution_kernel': str(ds.get('ConvolutionKernel', ''))
            })
        elif modality == 'MR':
            modality_info.update({
                'magnetic_field_strength': str(ds.get('MagneticFieldStrength', '')),
                'sequence_name': str(ds.get('SequenceName', '')),
                'repetition_time': str(ds.get('RepetitionTime', ''))
            })
        elif modality == 'PT':
            modality_info.update({
                'radiopharmaceutical': str(ds.get('RadiopharmaceuticalInformationSequence', [{}])[0].get('Radiopharmaceutical', '') if ds.get('RadiopharmaceuticalInformationSequence') else ''),
                'radionuclide_half_life': str(ds.get('RadiopharmaceuticalInformationSequence', [{}])[0].get('RadionuclideHalfLife', '') if ds.get('RadiopharmaceuticalInformationSequence') else ''),
                'decay_correction': str(ds.get('DecayCorrection', '')),
                'units': str(ds.get('Units', '')),
                'suv_type': str(ds.get('SUVType', ''))
            })
        
        return modality_info
    
    def validate_modality_support(self, ds: pydicom.Dataset) -> bool:
        """
        Valida se a modalidade é suportada pelo sistema
        
        Args:
            ds: Dataset DICOM
            
        Returns:
            True se modalidade é suportada
        """
        modality = str(ds.get('Modality', ''))
        is_supported = modality in self.SUPPORTED_MODALITIES
        
        if not is_supported:
            logger.warning(f"Modalidade não suportada: {modality}")
        else:
            logger.info(f"Modalidade suportada: {modality} - {self.SUPPORTED_MODALITIES[modality]}")
        
        return is_supported
