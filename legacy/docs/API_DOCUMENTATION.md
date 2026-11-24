# MedAI Radiologia API Documentation

## Overview

The MedAI Radiologia system provides a comprehensive REST API for medical image analysis using state-of-the-art AI models. The API supports multiple medical imaging modalities and provides detailed clinical validation and explainability features.

## Base URL

```
Production: https://your-domain.com/api/v1
Development: http://localhost:8000/api/v1
```

## Authentication

The API uses JWT (JSON Web Token) authentication for secure access.

### Headers Required
```http
Authorization: Bearer <your-jwt-token>
Content-Type: application/json
```

### Getting an Access Token
```bash
curl -X POST "https://your-domain.com/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your-username",
    "password": "your-password"
  }'
```

## Rate Limiting

- **Standard endpoints**: 60 requests per minute
- **Image analysis**: 10 requests per minute
- **Burst allowance**: 20 requests for standard, 5 for analysis

## API Endpoints

### 1. Health Check

**GET** `/health`

Check the system health and availability.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-11T22:56:59Z",
  "version": "4.0.0",
  "components": {
    "ai_models": "operational",
    "database": "connected",
    "storage": "available"
  },
  "uptime": 86400
}
```

### 2. List Available Models

**GET** `/models`

Retrieve information about available AI models.

**Response:**
```json
{
  "models": [
    {
      "name": "ensemble",
      "description": "Advanced ensemble model combining EfficientNetV2, Vision Transformer, and ConvNeXt",
      "modalities": ["CR", "CT", "MR", "US", "MG", "DX"],
      "accuracy": 0.94,
      "version": "4.0.0"
    },
    {
      "name": "efficientnetv2",
      "description": "EfficientNetV2 for general medical imaging",
      "modalities": ["CR", "DX"],
      "accuracy": 0.91,
      "version": "4.0.0"
    }
  ],
  "default_model": "ensemble"
}
```

### 3. Image Analysis

**POST** `/analyze`

Analyze a medical image using AI models.

**Request:**
```http
POST /api/v1/analyze
Content-Type: multipart/form-data
Authorization: Bearer <token>

Form Data:
- image: [file] (DICOM, JPEG, PNG)
- model: [string] (optional, default: "ensemble")
- include_explanation: [boolean] (optional, default: true)
- clinical_validation: [boolean] (optional, default: true)
```

**cURL Example:**
```bash
curl -X POST "https://your-domain.com/api/v1/analyze" \
  -H "Authorization: Bearer <your-token>" \
  -F "image=@chest_xray.dcm" \
  -F "model=ensemble" \
  -F "include_explanation=true"
```

**Response:**
```json
{
  "analysis": {
    "predicted_class": "Pneumonia",
    "confidence": 0.87,
    "findings": [
      "Consolidation in right lower lobe",
      "Increased opacity consistent with infection"
    ],
    "recommendations": [
      "Clinical correlation recommended",
      "Consider antibiotic therapy",
      "Follow-up imaging in 7-10 days"
    ],
    "severity": "moderate",
    "urgency": "routine"
  },
  "clinical_metrics": {
    "sensitivity": 0.92,
    "specificity": 0.89,
    "clinical_confidence": 0.85,
    "approved_for_clinical_use": true,
    "validation_notes": "Model performance within acceptable clinical thresholds"
  },
  "explanation": {
    "method": "gradcam",
    "heatmap_url": "https://your-domain.com/explanations/abc123.png",
    "key_regions": [
      {
        "region": "right_lower_lobe",
        "importance": 0.78,
        "coordinates": [120, 200, 180, 260]
      }
    ]
  },
  "metadata": {
    "model_used": "ensemble",
    "processing_time": 2.34,
    "image_info": {
      "modality": "CR",
      "dimensions": [512, 512],
      "file_size": "2.1 MB"
    },
    "timestamp": "2025-06-11T22:56:59Z",
    "request_id": "req_abc123def456"
  }
}
```

### 4. Get Analysis Explanation

**GET** `/explain/{analysis_id}`

Retrieve detailed explanation for a previous analysis.

**Parameters:**
- `analysis_id`: Unique identifier from previous analysis

**Response:**
```json
{
  "analysis_id": "req_abc123def456",
  "explanation": {
    "method": "gradcam",
    "heatmap_url": "https://your-domain.com/explanations/abc123.png",
    "integrated_gradients": {
      "attribution_map_url": "https://your-domain.com/attributions/abc123.png",
      "top_features": [
        {"feature": "lung_opacity", "importance": 0.78},
        {"feature": "consolidation_pattern", "importance": 0.65}
      ]
    },
    "clinical_interpretation": "The model focused primarily on areas of increased opacity in the right lower lobe, consistent with consolidation patterns typical of pneumonia."
  }
}
```

### 5. Performance Metrics

**GET** `/metrics`

Get system performance and clinical validation metrics.

**Response:**
```json
{
  "system_metrics": {
    "total_analyses": 15420,
    "average_processing_time": 2.1,
    "uptime_percentage": 99.8,
    "error_rate": 0.02
  },
  "clinical_metrics": {
    "overall_accuracy": 0.94,
    "sensitivity": 0.92,
    "specificity": 0.89,
    "positive_predictive_value": 0.87,
    "negative_predictive_value": 0.93
  },
  "model_performance": {
    "ensemble": {
      "accuracy": 0.94,
      "total_predictions": 12500,
      "clinical_approval_rate": 0.96
    },
    "efficientnetv2": {
      "accuracy": 0.91,
      "total_predictions": 2920,
      "clinical_approval_rate": 0.93
    }
  }
}
```

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_IMAGE_FORMAT",
    "message": "The uploaded file is not a valid medical image format",
    "details": "Supported formats: DICOM (.dcm), JPEG (.jpg, .jpeg), PNG (.png)",
    "timestamp": "2025-06-11T22:56:59Z",
    "request_id": "req_error_123"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_IMAGE_FORMAT` | 400 | Unsupported image format |
| `IMAGE_TOO_LARGE` | 413 | Image exceeds size limit (100MB) |
| `MODEL_NOT_FOUND` | 404 | Specified model doesn't exist |
| `PROCESSING_TIMEOUT` | 408 | Analysis took too long |
| `INSUFFICIENT_QUALITY` | 422 | Image quality too low for analysis |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `UNAUTHORIZED` | 401 | Invalid or missing authentication |
| `INTERNAL_ERROR` | 500 | Server processing error |

## SDKs and Code Examples

### Python SDK Example
```python
import requests
import json

class MedAIClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json'
        }
    
    def analyze_image(self, image_path, model='ensemble'):
        url = f"{self.base_url}/analyze"
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'model': model,
                'include_explanation': True,
                'clinical_validation': True
            }
            
            response = requests.post(
                url, 
                files=files, 
                data=data, 
                headers=self.headers
            )
            
        return response.json()

# Usage
client = MedAIClient('https://your-domain.com/api/v1', 'your-token')
result = client.analyze_image('chest_xray.dcm')
print(f"Diagnosis: {result['analysis']['predicted_class']}")
print(f"Confidence: {result['analysis']['confidence']:.2%}")
```

### JavaScript/Node.js Example
```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class MedAIClient {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${token}`,
            'Accept': 'application/json'
        };
    }
    
    async analyzeImage(imagePath, model = 'ensemble') {
        const form = new FormData();
        form.append('image', fs.createReadStream(imagePath));
        form.append('model', model);
        form.append('include_explanation', 'true');
        
        try {
            const response = await axios.post(
                `${this.baseUrl}/analyze`,
                form,
                {
                    headers: {
                        ...this.headers,
                        ...form.getHeaders()
                    }
                }
            );
            
            return response.data;
        } catch (error) {
            throw new Error(`Analysis failed: ${error.response?.data?.error?.message || error.message}`);
        }
    }
}

// Usage
const client = new MedAIClient('https://your-domain.com/api/v1', 'your-token');
client.analyzeImage('chest_xray.dcm')
    .then(result => {
        console.log(`Diagnosis: ${result.analysis.predicted_class}`);
        console.log(`Confidence: ${(result.analysis.confidence * 100).toFixed(1)}%`);
    })
    .catch(console.error);
```

## Webhooks

### Analysis Complete Webhook

Configure webhooks to receive notifications when analysis is complete.

**Webhook Payload:**
```json
{
  "event": "analysis.completed",
  "timestamp": "2025-06-11T22:56:59Z",
  "data": {
    "analysis_id": "req_abc123def456",
    "status": "completed",
    "predicted_class": "Pneumonia",
    "confidence": 0.87,
    "processing_time": 2.34,
    "clinical_approved": true
  }
}
```

## Best Practices

### 1. Image Quality
- Use high-resolution images (minimum 512x512 pixels)
- Ensure proper DICOM metadata is present
- Avoid heavily compressed images

### 2. Error Handling
- Always check the response status code
- Implement retry logic for transient errors
- Log request IDs for debugging

### 3. Performance
- Use appropriate model for your use case
- Consider batch processing for multiple images
- Cache results when appropriate

### 4. Security
- Store JWT tokens securely
- Use HTTPS in production
- Rotate tokens regularly

## Support and Contact

- **Documentation**: https://docs.medai-radiologia.com
- **Support Email**: support@medai-radiologia.com
- **Status Page**: https://status.medai-radiologia.com
- **GitHub Issues**: https://github.com/drguilhermecapel/radiologyai/issues

## Changelog

### v4.0.0 (2025-06-11)
- Added FastAPI REST API
- Implemented ensemble model support
- Added clinical validation framework
- Enhanced explainability features
- Multi-modality support (US, MG)

### v3.0.0 (2025-05-15)
- Initial REST API implementation
- Basic image analysis endpoints
- Authentication system
