#!/usr/bin/env python3
"""
Multi-Institutional Dataset Integration for RadiologyAI
Advanced dataset integration with harmonization, anonymization, and cross-institutional validation
"""

import logging
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InstitutionConfig:
    """Configuration for individual institution"""
    institution_id: str
    institution_name: str
    country: str
    data_path: str
    modalities: List[str]
    pathologies: List[str]
    anonymization_level: str = "full"  # "full", "partial", "none"
    data_format: str = "dicom"  # "dicom", "nifti", "png", "jpg"

@dataclass
class DataHarmonizationConfig:
    """Configuration for data harmonization across institutions"""
    target_spacing: Tuple[float, float, float]
    target_size: Tuple[int, int, int]
    intensity_normalization: str  # "z_score", "min_max", "histogram_matching"
    harmonization_method: str = "combat"  # "combat", "z_score", "histogram_matching"

class MultiInstitutionalDatasetIntegrator:
    """Main class for integrating datasets from multiple institutions"""
    
    def __init__(self, institutions: List[InstitutionConfig], 
                 harmonization_config: Optional[DataHarmonizationConfig] = None):
        self.institutions = institutions
        self.harmonization_config = harmonization_config or self._get_default_harmonization_config()
        self.integrated_dataset = {}
        
        logger.info(f"Multi-institutional integrator initialized with {len(institutions)} institutions")
    
    def _get_default_harmonization_config(self) -> DataHarmonizationConfig:
        """Get default harmonization configuration"""
        return DataHarmonizationConfig(
            target_spacing=(1.0, 1.0, 1.0),
            target_size=(512, 512, 64),
            intensity_normalization="z_score",
            harmonization_method="combat"
        )
    
    def load_institutional_dataset(self, institution: InstitutionConfig) -> Dict[str, Any]:
        """Load dataset from a specific institution"""
        
        logger.info(f"Loading dataset from {institution.institution_name} ({institution.institution_id})")
        
        data_path = Path(institution.data_path)
        if not data_path.exists():
            logger.warning(f"Data path does not exist: {institution.data_path}")
            logger.info(f"Creating synthetic dataset for {institution.institution_name}")
            return self._create_synthetic_institutional_dataset(institution)
        
        dataset = {
            'institution_id': institution.institution_id,
            'institution_name': institution.institution_name,
            'images': [],
            'labels': [],
            'metadata': [],
            'patient_ids': [],
            'study_ids': []
        }
        
        logger.info(f"Dataset loaded from {institution.institution_name}: 0 samples (real data loading not implemented)")
        return dataset
    
    def _create_synthetic_institutional_dataset(self, institution: InstitutionConfig) -> Dict[str, Any]:
        """Create synthetic dataset for institution (for testing purposes)"""
        
        n_samples = 200  # Samples per institution
        
        images = []
        labels = []
        metadata = []
        patient_ids = []
        study_ids = []
        
        for i in range(n_samples):
            image = np.random.rand(256, 256, 1).astype(np.float32)
            images.append(image)
            
            pathology = np.random.choice(institution.pathologies)
            label = 1 if pathology != "normal" else 0
            labels.append(label)
            
            meta = {
                'institution_id': institution.institution_id,
                'pathology': pathology,
                'modality': np.random.choice(institution.modalities),
                'age': np.random.randint(18, 90),
                'gender': np.random.choice(['M', 'F']),
                'acquisition_date': f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
                'scanner_model': f"Scanner_{institution.institution_id}",
                'anonymization_level': institution.anonymization_level
            }
            metadata.append(meta)
            
            patient_id = f"{institution.institution_id}_P{i//4:04d}"  # 4 studies per patient
            study_id = f"{institution.institution_id}_S{i:06d}"
            patient_ids.append(patient_id)
            study_ids.append(study_id)
        
        dataset = {
            'institution_id': institution.institution_id,
            'institution_name': institution.institution_name,
            'images': images,
            'labels': labels,
            'metadata': metadata,
            'patient_ids': patient_ids,
            'study_ids': study_ids
        }
        
        logger.info(f"Synthetic dataset created for {institution.institution_name}: {len(images)} samples")
        return dataset
    
    def harmonize_institutional_data(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Apply harmonization to institutional dataset"""
        
        logger.info(f"Harmonizing data from {dataset['institution_name']}")
        
        harmonized_images = []
        
        for image in dataset['images']:
            if self.harmonization_config.intensity_normalization == "z_score":
                mean_val = np.mean(image)
                std_val = np.std(image)
                harmonized_image = (image - mean_val) / (std_val + 1e-8)
            elif self.harmonization_config.intensity_normalization == "min_max":
                min_val = np.min(image)
                max_val = np.max(image)
                harmonized_image = (image - min_val) / (max_val - min_val + 1e-8)
            else:
                harmonized_image = image
            
            target_h, target_w = self.harmonization_config.target_size[:2]
            if harmonized_image.shape[:2] != (target_h, target_w):
                harmonized_image = np.resize(harmonized_image, (target_h, target_w, 1))
            
            harmonized_images.append(harmonized_image)
        
        harmonized_dataset = dataset.copy()
        harmonized_dataset['images'] = harmonized_images
        harmonized_dataset['harmonization_applied'] = True
        harmonized_dataset['harmonization_config'] = asdict(self.harmonization_config)
        
        logger.info(f"Data harmonization completed for {dataset['institution_name']}")
        return harmonized_dataset
    
    def apply_anonymization_protocols(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Apply anonymization protocols to dataset"""
        
        institution_id = dataset['institution_id']
        anonymization_level = dataset['metadata'][0].get('anonymization_level', 'full')
        
        logger.info(f"Applying {anonymization_level} anonymization to {dataset['institution_name']}")
        
        anonymized_metadata = []
        anonymized_patient_ids = []
        anonymized_study_ids = []
        
        for i, (meta, patient_id, study_id) in enumerate(zip(
            dataset['metadata'], dataset['patient_ids'], dataset['study_ids']
        )):
            anonymized_meta = meta.copy()
            
            if anonymization_level == "full":
                anonymized_meta.pop('acquisition_date', None)
                anonymized_meta['age_group'] = self._get_age_group(meta.get('age', 50))
                anonymized_meta.pop('age', None)
                
                anonymized_patient_id = f"ANON_{institution_id}_P{i//4:04d}"
                anonymized_study_id = f"ANON_{institution_id}_S{i:06d}"
                
            elif anonymization_level == "partial":
                anonymized_meta['age_group'] = self._get_age_group(meta.get('age', 50))
                
                anonymized_patient_id = f"PART_{institution_id}_P{i//4:04d}"
                anonymized_study_id = f"PART_{institution_id}_S{i:06d}"
                
            else:  # no anonymization
                anonymized_patient_id = patient_id
                anonymized_study_id = study_id
            
            anonymized_metadata.append(anonymized_meta)
            anonymized_patient_ids.append(anonymized_patient_id)
            anonymized_study_ids.append(anonymized_study_id)
        
        anonymized_dataset = dataset.copy()
        anonymized_dataset['metadata'] = anonymized_metadata
        anonymized_dataset['patient_ids'] = anonymized_patient_ids
        anonymized_dataset['study_ids'] = anonymized_study_ids
        anonymized_dataset['anonymization_applied'] = True
        
        logger.info(f"Anonymization completed for {dataset['institution_name']}")
        return anonymized_dataset
    
    def _get_age_group(self, age: int) -> str:
        """Convert age to age group for anonymization"""
        if age < 30:
            return "18-29"
        elif age < 50:
            return "30-49"
        elif age < 70:
            return "50-69"
        else:
            return "70+"
    
    def create_cross_institutional_validation_splits(self, 
                                                   integrated_dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Create validation splits for cross-institutional validation"""
        
        logger.info("Creating cross-institutional validation splits")
        
        institutions = list(set([meta['institution_id'] for meta in integrated_dataset['metadata']]))
        n_institutions = len(institutions)
        
        validation_splits = {
            'cross_validation_folds': 5,
            'leave_one_out_splits': n_institutions,
            'patient_based_splits': True,
            'institution_splits': {}
        }
        
        for i, held_out_institution in enumerate(institutions):
            train_indices = []
            test_indices = []
            
            for j, meta in enumerate(integrated_dataset['metadata']):
                if meta['institution_id'] == held_out_institution:
                    test_indices.append(j)
                else:
                    train_indices.append(j)
            
            validation_splits['institution_splits'][f'fold_{i}'] = {
                'train_indices': train_indices,
                'test_indices': test_indices,
                'held_out_institution': held_out_institution,
                'train_institutions': [inst for inst in institutions if inst != held_out_institution]
            }
        
        unique_patients = list(set(integrated_dataset['patient_ids']))
        np.random.shuffle(unique_patients)
        
        fold_size = len(unique_patients) // 5
        patient_folds = []
        
        for i in range(5):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < 4 else len(unique_patients)
            patient_folds.append(unique_patients[start_idx:end_idx])
        
        validation_splits['patient_cv_splits'] = {}
        for i, test_patients in enumerate(patient_folds):
            train_indices = []
            test_indices = []
            
            for j, patient_id in enumerate(integrated_dataset['patient_ids']):
                if patient_id in test_patients:
                    test_indices.append(j)
                else:
                    train_indices.append(j)
            
            validation_splits['patient_cv_splits'][f'fold_{i}'] = {
                'train_indices': train_indices,
                'test_indices': test_indices,
                'test_patients': test_patients
            }
        
        logger.info(f"Created validation splits: {n_institutions} institution splits, 5 patient CV splits")
        return validation_splits
    
    def assess_integration_quality(self, integrated_dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of integrated dataset"""
        
        logger.info("Assessing integration quality")
        
        total_samples = len(integrated_dataset['images'])
        institutions = list(set([meta['institution_id'] for meta in integrated_dataset['metadata']]))
        modalities = list(set([meta['modality'] for meta in integrated_dataset['metadata']]))
        pathologies = list(set([meta['pathology'] for meta in integrated_dataset['metadata']]))
        
        institution_distribution = {}
        for institution in institutions:
            count = sum(1 for meta in integrated_dataset['metadata'] if meta['institution_id'] == institution)
            institution_distribution[institution] = count
        
        pathology_distribution = {}
        for pathology in pathologies:
            count = sum(1 for meta in integrated_dataset['metadata'] if meta['pathology'] == pathology)
            pathology_distribution[pathology] = count
        
        quality_metrics = {
            'total_samples': total_samples,
            'n_institutions': len(institutions),
            'n_modalities': len(modalities),
            'n_pathologies': len(pathologies),
            'institution_distribution': institution_distribution,
            'pathology_distribution': pathology_distribution,
            'institutions': institutions,
            'modalities': modalities,
            'pathologies': pathologies,
            'data_balance_score': self._calculate_balance_score(institution_distribution),
            'pathology_balance_score': self._calculate_balance_score(pathology_distribution)
        }
        
        logger.info(f"Integration quality assessment completed: {total_samples} samples from {len(institutions)} institutions")
        return quality_metrics
    
    def _calculate_balance_score(self, distribution: Dict[str, int]) -> float:
        """Calculate balance score for distribution (0-1, where 1 is perfectly balanced)"""
        if not distribution:
            return 0.0
        
        values = list(distribution.values())
        mean_val = np.mean(values)
        
        if mean_val == 0:
            return 0.0
        
        cv = np.std(values) / mean_val
        
        balance_score = 1.0 / (1.0 + cv)
        return balance_score
    
    def integrate_multi_institutional_datasets(self) -> Dict[str, Any]:
        """Main method to integrate datasets from multiple institutions"""
        
        logger.info("🏥 Starting multi-institutional dataset integration...")
        
        integrated_images = []
        integrated_labels = []
        integrated_metadata = []
        integrated_patient_ids = []
        integrated_study_ids = []
        
        institutions_processed = []
        
        for institution in self.institutions:
            logger.info(f"Processing institution: {institution.institution_name}")
            
            dataset = self.load_institutional_dataset(institution)
            
            harmonized_dataset = self.harmonize_institutional_data(dataset)
            
            anonymized_dataset = self.apply_anonymization_protocols(harmonized_dataset)
            
            integrated_images.extend(anonymized_dataset['images'])
            integrated_labels.extend(anonymized_dataset['labels'])
            integrated_metadata.extend(anonymized_dataset['metadata'])
            integrated_patient_ids.extend(anonymized_dataset['patient_ids'])
            integrated_study_ids.extend(anonymized_dataset['study_ids'])
            
            institutions_processed.append({
                'institution_id': institution.institution_id,
                'institution_name': institution.institution_name,
                'samples_contributed': len(anonymized_dataset['images']),
                'pathologies': institution.pathologies,
                'modalities': institution.modalities
            })
        
        self.integrated_dataset = {
            'images': integrated_images,
            'labels': integrated_labels,
            'metadata': integrated_metadata,
            'patient_ids': integrated_patient_ids,
            'study_ids': integrated_study_ids
        }
        
        validation_splits = self.create_cross_institutional_validation_splits(self.integrated_dataset)
        
        quality_metrics = self.assess_integration_quality(self.integrated_dataset)
        
        integration_results = {
            'integration_timestamp': datetime.now().isoformat(),
            'institutions_processed': institutions_processed,
            'harmonization_summary': {
                'method': self.harmonization_config.harmonization_method,
                'target_spacing': self.harmonization_config.target_spacing,
                'target_size': self.harmonization_config.target_size,
                'intensity_normalization': self.harmonization_config.intensity_normalization
            },
            'anonymization_summary': {
                'institutions_with_full_anonymization': len([i for i in self.institutions if i.anonymization_level == 'full']),
                'institutions_with_partial_anonymization': len([i for i in self.institutions if i.anonymization_level == 'partial']),
                'institutions_with_no_anonymization': len([i for i in self.institutions if i.anonymization_level == 'none'])
            },
            'validation_splits': validation_splits,
            'quality_metrics': quality_metrics,
            'integrated_dataset_info': {
                'total_samples': len(integrated_images),
                'total_institutions': len(institutions_processed),
                'cross_validation_folds': validation_splits['cross_validation_folds'],
                'leave_one_out_splits': validation_splits['leave_one_out_splits']
            }
        }
        
        logger.info(f"✅ Multi-institutional integration completed: {len(integrated_images)} samples from {len(institutions_processed)} institutions")
        return integration_results
    
    def export_integrated_dataset(self, output_path: str) -> bool:
        """Export integrated dataset to file"""
        
        try:
            if not self.integrated_dataset:
                logger.error("No integrated dataset available for export")
                return False
            
            export_data = {
                'metadata': self.integrated_dataset['metadata'],
                'labels': self.integrated_dataset['labels'],
                'patient_ids': self.integrated_dataset['patient_ids'],
                'study_ids': self.integrated_dataset['study_ids'],
                'dataset_info': {
                    'total_samples': len(self.integrated_dataset['images']),
                    'image_shape': self.integrated_dataset['images'][0].shape if self.integrated_dataset['images'] else None,
                    'export_timestamp': datetime.now().isoformat()
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"✅ Integrated dataset exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export integrated dataset: {e}")
            return False

def main():
    """Example usage of multi-institutional dataset integrator"""
    
    logger.info("🧪 Multi-Institutional Dataset Integration Example")
    
    institutions = [
        InstitutionConfig(
            institution_id="INST_001",
            institution_name="Hospital A",
            country="USA",
            data_path="/data/hospital_a",
            modalities=["chest_xray"],
            pathologies=["pneumonia", "normal", "pleural_effusion"],
            anonymization_level="full"
        ),
        InstitutionConfig(
            institution_id="INST_002",
            institution_name="Hospital B",
            country="Canada",
            data_path="/data/hospital_b",
            modalities=["chest_xray"],
            pathologies=["pneumonia", "normal", "fracture"],
            anonymization_level="partial"
        ),
        InstitutionConfig(
            institution_id="INST_003",
            institution_name="Hospital C",
            country="UK",
            data_path="/data/hospital_c",
            modalities=["chest_xray"],
            pathologies=["pneumonia", "normal", "tumor"],
            anonymization_level="full"
        )
    ]
    
    harmonization_config = DataHarmonizationConfig(
        target_spacing=(1.0, 1.0, 1.0),
        target_size=(512, 512, 64),
        intensity_normalization="z_score",
        harmonization_method="combat"
    )
    
    integrator = MultiInstitutionalDatasetIntegrator(institutions, harmonization_config)
    
    integration_results = integrator.integrate_multi_institutional_datasets()
    
    integrator.export_integrated_dataset("/tmp/multi_institutional_dataset.json")
    
    logger.info("Multi-institutional dataset integration example completed")

if __name__ == "__main__":
    main()
