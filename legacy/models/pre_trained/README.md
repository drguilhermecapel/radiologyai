# Pre-trained Models Directory

This directory contains pre-trained AI models for the MedAI Radiologia system.

## Directory Structure

```
pre_trained/
├── efficientnetv2/          # EfficientNetV2 models
├── vision_transformer/      # Vision Transformer models  
├── convnext/               # ConvNeXt models
├── ensemble/               # Ensemble models
└── README.md              # This file
```

## Model Types

### EfficientNetV2 Models
- **Architecture**: EfficientNetV2-B3
- **Input Size**: 384x384x3
- **Specialization**: High accuracy chest X-ray analysis
- **Clinical Focus**: Pneumonia, pleural effusion, fractures, tumors

### Vision Transformer Models
- **Architecture**: ViT-Base/16
- **Input Size**: 224x224x3
- **Specialization**: Attention-based medical image analysis
- **Clinical Focus**: Complex pathology detection

### ConvNeXt Models
- **Architecture**: ConvNeXt-Base
- **Input Size**: 224x224x3
- **Specialization**: Modern convolutional architecture
- **Clinical Focus**: Balanced performance across modalities

### Ensemble Models
- **Architecture**: Multi-model ensemble
- **Input Size**: 384x384x3
- **Specialization**: Highest accuracy through model combination
- **Clinical Focus**: Multi-modal medical imaging

## Model Loading

Models are automatically downloaded and loaded by the `PreTrainedModelLoader` class. The system follows this priority:

1. **Local Pre-trained Models** (this directory)
2. **Download on Demand** (automatic download)
3. **Cloud Inference** (API-based)
4. **Basic Fallback** (lightweight models)

## Model Registry

The `model_registry.json` file contains metadata for all available models including:
- Download URLs and backup mirrors
- File integrity hashes (SHA256)
- Clinical validation information
- License and usage restrictions
- System requirements

## Usage

Models are loaded automatically by the system. No manual intervention required.

For developers:
```python
from medai_pretrained_loader import PreTrainedModelLoader

loader = PreTrainedModelLoader()
model = loader.load_pretrained_model("chest_xray_efficientnetv2")
```

## Legal Notice

All models are provided for research and educational purposes. Clinical use requires appropriate medical supervision and regulatory approval. See `MODELS_LICENSE.md` for detailed licensing information.
