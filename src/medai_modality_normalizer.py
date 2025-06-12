"""
MedAI Modality-Specific Normalizer - Advanced normalization for medical imaging
Implements proper Hounsfield Units conversion, bias field correction, and modality-specific windowing
"""

import numpy as np
import SimpleITK as sitk
import logging
from typing import Dict, Tuple, Optional, Union
import pydicom

logger = logging.getLogger('MedAI.ModalityNormalizer')

class ModalitySpecificNormalizer:
    """
    Advanced normalization system for medical imaging modalities
    Implements proper CT windowing with Hounsfield Units, MRI bias field correction,
    and modality-specific preprocessing as specified in the technical report
    """
    
    def __init__(self):
        self.ct_windows = {
            'lung': (-1000, -200),
            'mediastinum': (-175, 275), 
            'bone': (-500, 1300),
            'brain': (0, 80),
            'liver': (-150, 250),
            'soft_tissue': (-175, 275)
        }
        
        self.mri_sequences = {
            'T1': {'percentile_range': (1, 99)},
            'T2': {'percentile_range': (1, 99)},
            'FLAIR': {'percentile_range': (2, 98)},
            'DWI': {'percentile_range': (5, 95)}
        }
    
    def normalize_ct(self, dicom_data: pydicom.Dataset, target_organ: str = 'soft_tissue') -> np.ndarray:
        """
        Normalização específica para CT com janelamento HU
        Converte para Hounsfield Units e aplica janelamento específico por órgão
        """
        try:
            pixel_array = dicom_data.pixel_array.astype(float)
            
            if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                hu_image = pixel_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
            else:
                logger.warning("RescaleSlope/RescaleIntercept não encontrados, usando valores padrão")
                hu_image = pixel_array - 1024  # Valor padrão comum
            
            if target_organ in self.ct_windows:
                window_min, window_max = self.ct_windows[target_organ]
            else:
                window_min, window_max = self.ct_windows['soft_tissue']
            
            windowed = np.clip(hu_image, window_min, window_max)
            
            normalized = (windowed - window_min) / (window_max - window_min)
            
            logger.info(f"CT normalizado com janelamento {target_organ}: HU [{window_min}, {window_max}]")
            return normalized.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Erro na normalização CT: {e}")
            return self._fallback_normalization(dicom_data.pixel_array)
    
    def normalize_mri(self, image: np.ndarray, sequence_type: str = 'T1') -> np.ndarray:
        """
        Normalização para MRI com correção de bias field
        Implementa N4 bias field correction e normalização Z-score
        """
        try:
            if isinstance(image, np.ndarray):
                sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
            else:
                sitk_image = image
            
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
            corrector.SetConvergenceThreshold(1e-6)
            
            try:
                corrected = corrector.Execute(sitk_image)
                corrected_array = sitk.GetArrayFromImage(corrected)
            except:
                logger.warning("N4 bias correction falhou, usando imagem original")
                corrected_array = sitk.GetArrayFromImage(sitk_image)
            
            if sequence_type in self.mri_sequences:
                p_low, p_high = self.mri_sequences[sequence_type]['percentile_range']
            else:
                p_low, p_high = 1, 99
            
            threshold = np.percentile(corrected_array, p_low)
            mask = corrected_array > threshold
            
            if np.sum(mask) > 0:
                mean_val = np.mean(corrected_array[mask])
                std_val = np.std(corrected_array[mask])
                
                if std_val > 0:
                    normalized = (corrected_array - mean_val) / std_val
                    normalized = np.clip(normalized, -3, 3)
                    normalized = (normalized + 3) / 6
                else:
                    normalized = np.zeros_like(corrected_array)
            else:
                normalized = self._fallback_normalization(corrected_array)
            
            logger.info(f"MRI {sequence_type} normalizado com correção de bias field")
            return normalized.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Erro na normalização MRI: {e}")
            return self._fallback_normalization(image)
    
    def normalize_xray(self, image: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
        """
        Normalização específica para radiografias (CR/DX)
        Aplica CLAHE adaptativo e normalização robusta
        """
        try:
            if image.dtype != np.uint8:
                p1, p99 = np.percentile(image, [1, 99])
                if p99 > p1:
                    image_clipped = np.clip(image, p1, p99)
                    image_uint8 = ((image_clipped - p1) / (p99 - p1) * 255).astype(np.uint8)
                else:
                    image_uint8 = np.zeros_like(image, dtype=np.uint8)
            else:
                image_uint8 = image.copy()
            
            if enhance_contrast:
                import cv2
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(image_uint8)
            else:
                enhanced = image_uint8
            
            p1, p99 = np.percentile(enhanced, [1, 99])
            normalized = np.clip(enhanced, p1, p99)
            normalized = (normalized - p1) / (p99 - p1)
            
            logger.info("Radiografia normalizada com CLAHE adaptativo")
            return normalized.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Erro na normalização de radiografia: {e}")
            return self._fallback_normalization(image)
    
    def normalize_by_modality(self, image_data: Union[np.ndarray, pydicom.Dataset], 
                             modality: str, 
                             target_organ: str = 'soft_tissue',
                             sequence_type: str = 'T1') -> np.ndarray:
        """
        Normaliza imagem baseada na modalidade específica
        
        Args:
            image_data: Dados da imagem (array numpy ou dataset DICOM)
            modality: Modalidade (CT, MR, CR, DX, etc.)
            target_organ: Órgão alvo para CT windowing
            sequence_type: Tipo de sequência para MRI
            
        Returns:
            Imagem normalizada
        """
        try:
            if modality == 'CT':
                if isinstance(image_data, pydicom.Dataset):
                    return self.normalize_ct(image_data, target_organ)
                else:
                    logger.warning("CT normalization requires DICOM dataset, using fallback")
                    return self._fallback_normalization(image_data)
            
            elif modality in ['MR', 'MRI']:
                if isinstance(image_data, pydicom.Dataset):
                    image_array = image_data.pixel_array
                else:
                    image_array = image_data
                return self.normalize_mri(image_array, sequence_type)
            
            elif modality in ['CR', 'DX', 'CXR']:
                if isinstance(image_data, pydicom.Dataset):
                    image_array = image_data.pixel_array
                else:
                    image_array = image_data
                return self.normalize_xray(image_array)
            
            else:
                logger.warning(f"Modalidade não suportada: {modality}, usando normalização padrão")
                if isinstance(image_data, pydicom.Dataset):
                    image_array = image_data.pixel_array
                else:
                    image_array = image_data
                return self._fallback_normalization(image_array)
                
        except Exception as e:
            logger.error(f"Erro na normalização por modalidade {modality}: {e}")
            if isinstance(image_data, pydicom.Dataset):
                return self._fallback_normalization(image_data.pixel_array)
            else:
                return self._fallback_normalization(image_data)
    
    def _fallback_normalization(self, image: np.ndarray) -> np.ndarray:
        """Normalização de fallback segura"""
        try:
            image_min = np.percentile(image, 1)
            image_max = np.percentile(image, 99)
            
            if image_max > image_min:
                normalized = (image - image_min) / (image_max - image_min)
                return np.clip(normalized, 0, 1).astype(np.float32)
            else:
                return np.zeros_like(image, dtype=np.float32)
        except:
            return np.zeros_like(image, dtype=np.float32)

    def intelligent_resampling(self, image: sitk.Image, target_spacing: list = [1.0, 1.0, 1.0]) -> sitk.Image:
        """
        Resampling preservando informação diagnóstica
        Usa interpolação B-spline para manter qualidade
        """
        try:
            original_spacing = image.GetSpacing()
            original_size = image.GetSize()
            
            new_size = [
                int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
                for i in range(len(original_size))
            ]
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(target_spacing)
            resampler.SetSize(new_size)
            resampler.SetInterpolator(sitk.sitkBSpline)  # Superior para imagens médicas
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetOutputOrigin(image.GetOrigin())
            
            resampled = resampler.Execute(image)
            logger.info(f"Resampling inteligente aplicado: {original_spacing} -> {target_spacing}")
            return resampled
            
        except Exception as e:
            logger.error(f"Erro no resampling inteligente: {e}")
            return image
