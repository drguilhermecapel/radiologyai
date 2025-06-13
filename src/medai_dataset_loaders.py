#!/usr/bin/env python3
"""
Medical Dataset Loaders for Clinical Training
Supports CheXpert, LIDC-IDRI, BraTS, NIH ChestX-ray, and synthetic datasets
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MedicalDatasetLoaders')

class BaseDatasetLoader(ABC):
    """Base class for medical dataset loaders"""
    
    def __init__(self, base_data_dir: str):
        self.base_data_dir = Path(base_data_dir)
        self.dataset_dir = None
        self.dataset_info = {}
        
    @abstractmethod
    def download_dataset(self) -> bool:
        """Download the dataset if not available"""
        pass
        
    @abstractmethod
    def load_dataset(self) -> Tuple[List, List, List, List, List, List]:
        """Load dataset and return train/val/test splits"""
        pass
        
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        pass
        
    def validate_dataset(self) -> bool:
        """Validate dataset integrity"""
        if not self.dataset_dir or not self.dataset_dir.exists():
            logger.error(f"Dataset directory not found: {self.dataset_dir}")
            return False
        return True
        
    def create_balanced_splits(self, files: List, labels: List, 
                             train_ratio: float = 0.7, 
                             val_ratio: float = 0.15) -> Tuple[List, List, List, List, List, List]:
        """Create balanced train/validation/test splits"""
        
        files_array = np.array(files)
        labels_array = np.array(labels)
        
        unique_classes = np.unique(labels_array)
        
        train_files, train_labels = [], []
        val_files, val_labels = [], []
        test_files, test_labels = [], []
        
        for class_label in unique_classes:
            class_indices = np.where(labels_array == class_label)[0]
            np.random.shuffle(class_indices)
            
            n_samples = len(class_indices)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            train_indices = class_indices[:n_train]
            val_indices = class_indices[n_train:n_train + n_val]
            test_indices = class_indices[n_train + n_val:]
            
            train_files.extend(files_array[train_indices].tolist())
            train_labels.extend(labels_array[train_indices].tolist())
            
            val_files.extend(files_array[val_indices].tolist())
            val_labels.extend(labels_array[val_indices].tolist())
            
            test_files.extend(files_array[test_indices].tolist())
            test_labels.extend(labels_array[test_indices].tolist())
        
        return train_files, train_labels, val_files, val_labels, test_files, test_labels

class SyntheticChestXrayLoader(BaseDatasetLoader):
    """Loader for synthetic chest X-ray dataset"""
    
    def __init__(self, base_data_dir: str):
        super().__init__(base_data_dir)
        self.dataset_dir = self.base_data_dir / "synthetic_chest_xray"
        self.dataset_info = {
            'name': 'Synthetic Chest X-ray',
            'modality': 'X-ray',
            'classes': ['normal', 'pneumonia', 'covid19', 'tuberculosis', 
                       'lung_cancer', 'pneumothorax', 'pleural_effusion', 'atelectasis'],
            'total_samples': 12000,
            'image_size': (224, 224),
            'format': 'PNG'
        }
        
    def download_dataset(self) -> bool:
        """Synthetic dataset is generated locally"""
        if not self.dataset_dir.exists():
            logger.info("Synthetic dataset not found. Please run create_synthetic_medical_dataset.py")
            return False
        return True
        
    def load_dataset(self) -> Tuple[List, List, List, List, List, List]:
        """Load synthetic chest X-ray dataset"""
        
        if not self.validate_dataset():
            logger.error("Dataset validation failed")
            return [], [], [], [], [], []
            
        train_labels_file = self.dataset_dir / "train_labels.json"
        val_labels_file = self.dataset_dir / "val_labels.json"
        test_labels_file = self.dataset_dir / "test_labels.json"
        
        if not all([f.exists() for f in [train_labels_file, val_labels_file, test_labels_file]]):
            logger.error("Label files not found")
            return [], [], [], [], [], []
            
        with open(train_labels_file, 'r') as f:
            train_data = json.load(f)
        with open(val_labels_file, 'r') as f:
            val_data = json.load(f)
        with open(test_labels_file, 'r') as f:
            test_data = json.load(f)
            
        train_files = [str(self.dataset_dir / f) for f in train_data['files']]
        train_labels = train_data['labels']
        
        val_files = [str(self.dataset_dir / f) for f in val_data['files']]
        val_labels = val_data['labels']
        
        test_files = [str(self.dataset_dir / f) for f in test_data['files']]
        test_labels = test_data['labels']
        
        logger.info(f"Loaded synthetic dataset: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        return train_files, train_labels, val_files, val_labels, test_files, test_labels
        
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get synthetic dataset information"""
        return self.dataset_info

class CheXpertDatasetLoader(BaseDatasetLoader):
    """Loader for CheXpert dataset"""
    
    def __init__(self, base_data_dir: str):
        super().__init__(base_data_dir)
        self.dataset_dir = self.base_data_dir / "chexpert"
        self.dataset_info = {
            'name': 'CheXpert',
            'modality': 'X-ray',
            'classes': ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
                       'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                       'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                       'Pleural Other', 'Fracture', 'Support Devices'],
            'total_samples': 224316,
            'image_size': (320, 320),
            'format': 'JPG'
        }
        
    def download_dataset(self) -> bool:
        """Download CheXpert dataset"""
        logger.info("CheXpert dataset requires manual download from Stanford")
        logger.info("Please visit: https://stanfordmlgroup.github.io/competitions/chexpert/")
        return self.dataset_dir.exists()
        
    def load_dataset(self) -> Tuple[List, List, List, List, List, List]:
        """Load CheXpert dataset"""
        logger.warning("CheXpert loader not fully implemented - requires manual dataset download")
        return [], [], [], [], [], []
        
    def get_dataset_info(self) -> Dict[str, Any]:
        return self.dataset_info

class LIDCIDRIDatasetLoader(BaseDatasetLoader):
    """Loader for LIDC-IDRI dataset"""
    
    def __init__(self, base_data_dir: str):
        super().__init__(base_data_dir)
        self.dataset_dir = self.base_data_dir / "lidc_idri"
        self.dataset_info = {
            'name': 'LIDC-IDRI',
            'modality': 'CT',
            'classes': ['benign', 'malignant'],
            'total_samples': 1018,
            'image_size': (512, 512),
            'format': 'DICOM'
        }
        
    def download_dataset(self) -> bool:
        """Download LIDC-IDRI dataset"""
        logger.info("LIDC-IDRI dataset requires TCIA access")
        logger.info("Please visit: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI")
        return self.dataset_dir.exists()
        
    def load_dataset(self) -> Tuple[List, List, List, List, List, List]:
        """Load LIDC-IDRI dataset"""
        logger.warning("LIDC-IDRI loader not fully implemented - requires TCIA access")
        return [], [], [], [], [], []
        
    def get_dataset_info(self) -> Dict[str, Any]:
        return self.dataset_info

class BraTSDatasetLoader(BaseDatasetLoader):
    """Loader for BraTS dataset"""
    
    def __init__(self, base_data_dir: str):
        super().__init__(base_data_dir)
        self.dataset_dir = self.base_data_dir / "brats"
        self.dataset_info = {
            'name': 'BraTS',
            'modality': 'MRI',
            'classes': ['background', 'necrotic_core', 'peritumoral_edema', 'enhancing_tumor'],
            'total_samples': 369,
            'image_size': (240, 240, 155),
            'format': 'NIfTI'
        }
        
    def download_dataset(self) -> bool:
        """Download BraTS dataset"""
        logger.info("BraTS dataset requires registration")
        logger.info("Please visit: http://braintumorsegmentation.org/")
        return self.dataset_dir.exists()
        
    def load_dataset(self) -> Tuple[List, List, List, List, List, List]:
        """Load BraTS dataset"""
        logger.warning("BraTS loader not fully implemented - requires registration")
        return [], [], [], [], [], []
        
    def get_dataset_info(self) -> Dict[str, Any]:
        return self.dataset_info

class NIHChestXrayLoader(BaseDatasetLoader):
    """Loader for NIH ChestX-ray14 dataset"""
    
    def __init__(self, base_data_dir: str):
        super().__init__(base_data_dir)
        self.dataset_dir = self.base_data_dir / "nih_chest_xray"
        self.dataset_info = {
            'name': 'NIH ChestX-ray14',
            'modality': 'X-ray',
            'classes': ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                       'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
                       'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'],
            'total_samples': 112120,
            'image_size': (1024, 1024),
            'format': 'PNG'
        }
        
    def download_dataset(self) -> bool:
        """Download NIH ChestX-ray14 dataset"""
        logger.info("NIH ChestX-ray14 dataset requires download from NIH")
        logger.info("Please visit: https://nihcc.app.box.com/v/ChestXray-NIHCC")
        return self.dataset_dir.exists()
        
    def load_dataset(self) -> Tuple[List, List, List, List, List, List]:
        """Load NIH ChestX-ray14 dataset"""
        logger.warning("NIH ChestX-ray14 loader not fully implemented - requires manual download")
        return [], [], [], [], [], []
        
    def get_dataset_info(self) -> Dict[str, Any]:
        return self.dataset_info

class MedicalDatasetManager:
    """Manager for multiple medical dataset loaders"""
    
    def __init__(self, base_data_dir: str):
        self.base_data_dir = base_data_dir
        self.loaders = {
            'synthetic_chest_xray': SyntheticChestXrayLoader,
            'chexpert': CheXpertDatasetLoader,
            'lidc_idri': LIDCIDRIDatasetLoader,
            'brats': BraTSDatasetLoader,
            'nih_chest_xray': NIHChestXrayLoader
        }
        
    def get_loader(self, dataset_name: str) -> BaseDatasetLoader:
        """Get a dataset loader by name"""
        if dataset_name not in self.loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        return self.loaders[dataset_name](self.base_data_dir)
        
    def list_available_datasets(self) -> List[str]:
        """List all available dataset loaders"""
        return list(self.loaders.keys())
        
    def validate_all_datasets(self) -> Dict[str, bool]:
        """Validate all available datasets"""
        results = {}
        for dataset_name in self.loaders:
            try:
                loader = self.get_loader(dataset_name)
                results[dataset_name] = loader.validate_dataset()
            except Exception as e:
                logger.error(f"Error validating {dataset_name}: {e}")
                results[dataset_name] = False
                
        return results

if __name__ == "__main__":
    manager = MedicalDatasetManager("data/medical_datasets")
    
    print("Available datasets:", manager.list_available_datasets())
    
    try:
        synthetic_loader = manager.get_loader('synthetic_chest_xray')
        dataset_info = synthetic_loader.get_dataset_info()
        print(f"Synthetic dataset info: {dataset_info}")
        
        train_files, train_labels, val_files, val_labels, test_files, test_labels = synthetic_loader.load_dataset()
        print(f"Loaded: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test samples")
        
    except Exception as e:
        print(f"Error testing synthetic dataset: {e}")
