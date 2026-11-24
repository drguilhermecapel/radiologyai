#!/usr/bin/env python3
"""
MedAI Population Adaptation System
Adapts AI models for different demographic groups and populations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

class PopulationGroup(Enum):
    """Enumeration of population groups for adaptation"""
    PEDIATRIC = "pediatric"
    ADULT = "adult"
    GERIATRIC = "geriatric"
    MALE = "male"
    FEMALE = "female"
    ASIAN = "asian"
    CAUCASIAN = "caucasian"
    AFRICAN = "african"
    HISPANIC = "hispanic"
    LOW_BMI = "low_bmi"
    HIGH_BMI = "high_bmi"

@dataclass
class PopulationCharacteristics:
    """Characteristics of a specific population group"""
    group_id: str
    demographic_factors: Dict[str, Any]
    anatomical_variations: Dict[str, float]
    disease_prevalence: Dict[str, float]
    imaging_parameters: Dict[str, Any]
    model_adjustments: Dict[str, float]

@dataclass
class AdaptationResult:
    """Results from population adaptation"""
    original_prediction: np.ndarray
    adapted_prediction: np.ndarray
    adaptation_factors: Dict[str, float]
    confidence_adjustment: float
    population_groups: List[PopulationGroup]
    adaptation_timestamp: Any

class PopulationModelAdapter:
    """Adapts AI models for different population groups"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.population_characteristics = self._load_population_data()
        self.adaptation_models = {}
        self.logger = logging.getLogger(__name__)
        
    def _load_population_data(self) -> Dict[PopulationGroup, PopulationCharacteristics]:
        """Load population-specific characteristics and adaptations"""
        
        characteristics = {}
        
        characteristics[PopulationGroup.PEDIATRIC] = PopulationCharacteristics(
            group_id="pediatric",
            demographic_factors={"age_range": (0, 18), "growth_stage": "developing"},
            anatomical_variations={"organ_size_ratio": 0.7, "bone_density": 0.8},
            disease_prevalence={"pneumonia": 0.15, "fractures": 0.25},
            imaging_parameters={"dose_reduction": 0.5, "contrast_adjustment": 0.8},
            model_adjustments={"sensitivity_boost": 1.2, "specificity_adjustment": 0.95}
        )
        
        characteristics[PopulationGroup.GERIATRIC] = PopulationCharacteristics(
            group_id="geriatric",
            demographic_factors={"age_range": (65, 100), "comorbidity_rate": 0.8},
            anatomical_variations={"bone_density": 0.6, "organ_atrophy": 1.3},
            disease_prevalence={"osteoporosis": 0.4, "cardiovascular": 0.6},
            imaging_parameters={"contrast_caution": True, "acquisition_time": 1.2},
            model_adjustments={"age_factor": 1.15, "comorbidity_weight": 1.3}
        )
        
        return characteristics
    
    def identify_population_groups(self, patient_metadata: Dict[str, Any]) -> List[PopulationGroup]:
        """Identify applicable population groups for a patient"""
        groups = []
        
        age = patient_metadata.get('age', 0)
        if age < 18:
            groups.append(PopulationGroup.PEDIATRIC)
        elif age >= 65:
            groups.append(PopulationGroup.GERIATRIC)
        else:
            groups.append(PopulationGroup.ADULT)
        
        gender = patient_metadata.get('gender', '').lower()
        if gender == 'female':
            groups.append(PopulationGroup.FEMALE)
        elif gender == 'male':
            groups.append(PopulationGroup.MALE)
        
        return groups
    
    def adapt_prediction(self, original_prediction: np.ndarray,
                        patient_metadata: Dict[str, Any],
                        imaging_metadata: Dict[str, Any]) -> AdaptationResult:
        """Adapt AI prediction for specific population characteristics"""
        
        population_groups = self.identify_population_groups(patient_metadata)
        
        if not population_groups:
            return AdaptationResult(
                original_prediction=original_prediction,
                adapted_prediction=original_prediction,
                adaptation_factors={},
                confidence_adjustment=1.0,
                population_groups=[],
                adaptation_timestamp=pd.Timestamp.now()
            )
        
        adapted_prediction = original_prediction.copy()
        adaptation_factors = {}
        confidence_adjustment = 1.0
        
        for group in population_groups:
            if group in self.population_characteristics:
                char = self.population_characteristics[group]
                
                for adjustment_type, factor in char.model_adjustments.items():
                    if adjustment_type == 'sensitivity_boost':
                        adapted_prediction *= factor
                        adaptation_factors[f"{group.value}_sensitivity"] = factor
                    elif adjustment_type == 'age_factor':
                        adapted_prediction = self._apply_age_adjustment(
                            adapted_prediction, patient_metadata.get('age', 0), factor
                        )
                        adaptation_factors[f"{group.value}_age"] = factor
                
                confidence_adjustment *= self._calculate_confidence_adjustment(char, patient_metadata)
        
        adapted_prediction = np.clip(adapted_prediction, 0, 1)
        
        return AdaptationResult(
            original_prediction=original_prediction,
            adapted_prediction=adapted_prediction,
            adaptation_factors=adaptation_factors,
            confidence_adjustment=confidence_adjustment,
            population_groups=population_groups,
            adaptation_timestamp=pd.Timestamp.now()
        )
    
    def _apply_age_adjustment(self, prediction: np.ndarray, age: int, factor: float) -> np.ndarray:
        """Apply age-specific adjustments"""
        age_weight = 1.0 + (age - 50) / 100 * (factor - 1.0)
        return prediction * age_weight
    
    def _calculate_confidence_adjustment(self, characteristics: PopulationCharacteristics,
                                       patient_metadata: Dict[str, Any]) -> float:
        """Calculate confidence adjustment based on population characteristics"""
        base_confidence = 1.0
        
        if characteristics.group_id == 'pediatric':
            base_confidence *= 0.9
        elif characteristics.group_id == 'geriatric':
            base_confidence *= 0.95
        
        return base_confidence

class PopulationAdaptationSystem:
    """Main system for population-specific AI adaptation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.model_adapter = PopulationModelAdapter(self.config)
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            'adaptation_enabled': True,
            'confidence_threshold': 0.8,
            'adaptation_logging': True
        }
        
        if config_path and Path(config_path).exists():
            pass
            
        return default_config
    
    def process_with_adaptation(self, prediction: np.ndarray,
                              patient_metadata: Dict[str, Any],
                              imaging_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process prediction with population adaptation"""
        
        if not self.config.get('adaptation_enabled', True):
            return {
                'original_prediction': prediction,
                'adapted_prediction': prediction,
                'adaptation_applied': False
            }
        
        adaptation_result = self.model_adapter.adapt_prediction(
            prediction, patient_metadata, imaging_metadata
        )
        
        return {
            'original_prediction': adaptation_result.original_prediction,
            'adapted_prediction': adaptation_result.adapted_prediction,
            'adaptation_factors': adaptation_result.adaptation_factors,
            'confidence_adjustment': adaptation_result.confidence_adjustment,
            'population_groups': [g.value for g in adaptation_result.population_groups],
            'adaptation_applied': True,
            'adaptation_timestamp': adaptation_result.adaptation_timestamp
        }

def main():
    """Example usage of population adaptation system"""
    
    system = PopulationAdaptationSystem()
    
    prediction = np.array([0.1, 0.3, 0.8, 0.2, 0.1])
    patient_metadata = {
        'age': 75,
        'gender': 'female',
        'ethnicity': 'asian',
        'bmi': 22.5
    }
    imaging_metadata = {
        'modality': 'CT',
        'contrast': True
    }
    
    result = system.process_with_adaptation(
        prediction, patient_metadata, imaging_metadata
    )
    
    print("Population Adaptation Results:")
    print(f"Original prediction: {result['original_prediction']}")
    print(f"Adapted prediction: {result['adapted_prediction']}")
    print(f"Population groups: {result['population_groups']}")

if __name__ == "__main__":
    main()
