# test_system.py - Sistema completo de testes e validação

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
import shutil
import json
import yaml
from datetime import datetime
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileDataset
import cv2
import h5py
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import psutil
import GPUtil
from dataclasses import dataclass
import warnings

# Importar módulos do sistema para teste
from dicom_processor import DICOMProcessor
from neural_networks import MedicalImageNetwork
from training_system import MedicalModelTrainer
from inference_system import MedicalInferenceEngine, PredictionResult
from batch_processor import BatchProcessor, BatchJob
from comparison_system import ImageComparisonSystem, ComparisonResult
from security_audit import SecurityManager, UserRole, AuditEventType
from export_system import ExportManager

# Configurar logging para testes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MedAI.Tests')

# Fixtures e utilitários
@pytest.fixture
def temp_dir():
    """Cria diretório temporário para testes"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def sample_dicom():
    """Cria DICOM de exemplo para testes"""
    # Criar dataset DICOM mínimo
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = '1.2.3.4.5.6.7.8.9'
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
    
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Informações obrigatórias
    ds.PatientName = 'Test^Patient'
    ds.PatientID = '123456'
    ds.StudyDate = '20240101'
    ds.StudyTime = '120000'
    ds.Modality = 'CT'
    ds.StudyInstanceUID = '1.2.3.4.5'
    ds.SeriesInstanceUID = '1.2.3.4.5.6'
    ds.SOPInstanceUID = '1.2.3.4.5.6.7'
    
    # Dados de pixel
    ds.Rows = 256
    ds.Columns = 256
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    
    # Criar array de pixels
    pixel_array = np.random.randint(0, 4096, (256, 256), dtype=np.uint16)
    ds.PixelData = pixel_array.tobytes()
    
    return ds

@pytest.fixture
def sample_image():
    """Cria imagem de exemplo para testes"""
    # Criar imagem sintética com padrão
    image = np.zeros((224, 224), dtype=np.uint8)
    
    # Adicionar alguns elementos
    cv2.circle(image, (112, 112), 50, 255, -1)  # Círculo central
    cv2.rectangle(image, (50, 50), (174, 174), 128, 2)  # Retângulo
    
    # Adicionar ruído
    noise = np.random.normal(0, 10, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return image

@pytest.fixture
def sample_model(temp_dir):
    """Cria modelo de exemplo para testes"""
    # Criar modelo simples
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Salvar modelo
    model_path = Path(temp_dir) / 'test_model.h5'
    model.save(model_path)
    
    return model_path

# Testes Unitários
class TestDICOMProcessor(unittest.TestCase):
    """Testes para processador DICOM"""
    
    def setUp(self):
        self.processor = DICOMProcessor(anonymize=True)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_read_dicom(self):
        """Testa leitura de arquivo DICOM"""
        # Criar DICOM temporário
        ds = sample_dicom()
        dicom_path = Path(self.temp_dir) / 'test.dcm'
        ds.save_as(str(dicom_path))
        
        # Ler DICOM
        loaded_ds = self.processor.read_dicom(dicom_path)
        
        # Verificar
        self.assertIsNotNone(loaded_ds)
        self.assertEqual(loaded_ds.Rows, 256)
        self.assertEqual(loaded_ds.Columns, 256)
    
    def test_anonymization(self):
        """Testa anonimização de dados sensíveis"""
        ds = sample_dicom()
        
        # Anonimizar
        anon_ds = self.processor._anonymize_dicom(ds)
        
        # Verificar
        self.assertNotEqual(anon_ds.PatientName, ds.PatientName)
        self.assertTrue(anon_ds.PatientName.startswith('ANON_'))
        self.assertNotEqual(anon_ds.PatientID, ds.PatientID)
        self.assertTrue(anon_ds.PatientID.startswith('ID_'))
    
    def test_dicom_to_array(self):
        """Testa conversão DICOM para array"""
        ds = sample_dicom()
        
        # Adicionar transformações
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024
        ds.WindowCenter = 40
        ds.WindowWidth = 400
        
        # Converter
        array = self.processor.dicom_to_array(ds)
        
        # Verificar
        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.shape, (256, 256))
        self.assertEqual(array.dtype, np.uint8)
        self.assertTrue(0 <= array.min() <= array.max() <= 255)
    
    def test_extract_metadata(self):
        """Testa extração de metadados"""
        ds = sample_dicom()
        
        # Extrair metadados
        metadata = self.processor.extract_metadata(ds)
        
        # Verificar campos essenciais
        self.assertIn('StudyDate', metadata)
        self.assertIn('Modality', metadata)
        self.assertIn('Rows', metadata)
        self.assertIn('Columns', metadata)
        
        # Verificar anonimização
        self.processor.anonymize = True
        metadata_anon = self.processor.extract_metadata(ds)
        self.assertNotIn('PatientName', metadata_anon)
        self.assertNotIn('PatientID', metadata_anon)
    
    def test_preprocess_for_ai(self):
        """Testa pré-processamento para IA"""
        image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        
        # Pré-processar
        processed = self.processor.preprocess_for_ai(
            image, 
            target_size=(224, 224),
            normalize=True
        )
        
        # Verificar
        self.assertEqual(processed.shape, (224, 224, 1))
        self.assertTrue(0 <= processed.min() <= processed.max() <= 1)
        self.assertEqual(processed.dtype, np.float32)

class TestNeuralNetworks(unittest.TestCase):
    """Testes para redes neurais"""
    
    def test_model_creation(self):
        """Testa criação de diferentes arquiteturas"""
        architectures = ['densenet', 'resnet', 'efficientnet', 'custom_cnn']
        
        for arch in architectures:
            network = MedicalImageNetwork(
                input_shape=(224, 224, 3),
                num_classes=5,
                model_name=arch
            )
            
            model = network.build_model()
            
            # Verificar estrutura
            self.assertIsNotNone(model)
            self.assertEqual(model.input_shape[1:], (224, 224, 3))
            self.assertEqual(model.output_shape[1], 5)
    
    def test_model_compilation(self):
        """Testa compilação do modelo"""
        network = MedicalImageNetwork(
            input_shape=(224, 224, 1),
            num_classes=3,
            model_name='custom_cnn'
        )
        
        model = network.build_model()
        network.compile_model(learning_rate=0.001)
        
        # Verificar compilação
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
        self.assertTrue(len(model.metrics) > 0)
    
    def test_attention_unet(self):
        """Testa U-Net com atenção"""
        network = MedicalImageNetwork(
            input_shape=(256, 256, 1),
            num_classes=2,
            model_name='attention_unet'
        )
        
        model = network.build_model()
        
        # Verificar saída de segmentação
        self.assertEqual(model.output_shape[1:], (256, 256, 2))
    
    @pytest.mark.slow
    def test_model_inference(self):
        """Testa inferência do modelo"""
        network = MedicalImageNetwork(
            input_shape=(224, 224, 1),
            num_classes=5,
            model_name='custom_cnn'
        )
        
        model = network.build_model()
        network.compile_model()
        
        # Criar batch de teste
        batch = np.random.random((4, 224, 224, 1)).astype(np.float32)
        
        # Inferência
        predictions = model.predict(batch)
        
        # Verificar
        self.assertEqual(predictions.shape, (4, 5))
        self.assertTrue(np.allclose(predictions.sum(axis=1), 1.0, rtol=1e-5))

class TestInferenceSystem(unittest.TestCase):
    """Testes para sistema de inferência"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Criar modelo mock
        self.model_config = {
            'classes': ['Normal', 'Pneumonia', 'COVID-19'],
            'input_size': (224, 224),
            'threshold': 0.5
        }
        
        # Mock do modelo
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([[0.7, 0.2, 0.1]])
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('tensorflow.keras.models.load_model')
    def test_predict_single(self, mock_load_model):
        """Testa predição única"""
        mock_load_model.return_value = self.mock_model
        
        # Criar engine
        engine = MedicalInferenceEngine(
            Path(self.temp_dir) / 'model.h5',
            self.model_config
        )
        engine.model = self.mock_model
        
        # Criar imagem de teste
        image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        # Predição
        result = engine.predict_single(image, return_attention=False)
        
        # Verificar
        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.predicted_class, 'Normal')
        self.assertAlmostEqual(result.confidence, 0.7)
        self.assertEqual(len(result.predictions), 3)
    
    def test_analyze_uncertainty(self):
        """Testa análise de incerteza"""
        engine = MedicalInferenceEngine(
            Path(self.temp_dir) / 'model.h5',
            self.model_config
        )
        
        # Predições de teste
        predictions = np.array([0.8, 0.15, 0.05])
        
        # Analisar incerteza
        uncertainty = engine.analyze_uncertainty(predictions, 3)
        
        # Verificar métricas
        self.assertIn('entropy', uncertainty)
        self.assertIn('normalized_entropy', uncertainty)
        self.assertIn('margin', uncertainty)
        self.assertIn('variance', uncertainty)
        self.assertIn('max_probability', uncertainty)
        
        # Verificar valores
        self.assertTrue(0 <= uncertainty['normalized_entropy'] <= 1)
        self.assertAlmostEqual(uncertainty['margin'], 0.65)  # 0.8 - 0.15
        self.assertEqual(uncertainty['max_probability'], 0.8)

class TestBatchProcessor(unittest.TestCase):
    """Testes para processamento em lote"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Criar arquivos de teste
        self.test_files = []
        for i in range(10):
            img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
            img_path = Path(self.temp_dir) / f'test_{i}.png'
            cv2.imwrite(str(img_path), img)
            self.test_files.append(img_path)
        
        # Mock do inference engine
        self.mock_engine = MagicMock()
        self.mock_engine.predict_single.return_value = PredictionResult(
            image_path='',
            predictions={'Normal': 0.8, 'Abnormal': 0.2},
            predicted_class='Normal',
            confidence=0.8,
            processing_time=0.1
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_create_batch_job(self):
        """Testa criação de job em lote"""
        processor = BatchProcessor(self.mock_engine)
        
        # Criar job
        job = processor.create_batch_job(
            [str(f) for f in self.test_files],
            self.temp_dir,
            'test_model'
        )
        
        # Verificar
        self.assertIsInstance(job, BatchJob)
        self.assertEqual(len(job.input_files), 10)
        self.assertEqual(job.status, 'pending')
        self.assertEqual(job.model_type, 'test_model')
    
    def test_process_batch(self):
        """Testa processamento em lote"""
        processor = BatchProcessor(
            self.mock_engine,
            max_workers=2,
            batch_size=5
        )
        
        # Criar job pequeno
        job = processor.create_batch_job(
            [str(f) for f in self.test_files[:5]],
            self.temp_dir,
            'test_model'
        )
        
        # Processar
        summary = processor.process_batch(job)
        
        # Verificar
        self.assertEqual(job.status, 'completed')
        self.assertEqual(len(job.results), 5)
        self.assertEqual(summary['processed'], 5)
        self.assertEqual(summary['failed'], 0)
    
    def test_batch_error_handling(self):
        """Testa tratamento de erros em lote"""
        # Configurar mock para falhar em alguns arquivos
        self.mock_engine.predict_single.side_effect = [
            PredictionResult('', {}, 'Normal', 0.8, 0.1),
            Exception("Erro de teste"),
            PredictionResult('', {}, 'Normal', 0.8, 0.1),
        ]
        
        processor = BatchProcessor(self.mock_engine)
        
        job = processor.create_batch_job(
            [str(f) for f in self.test_files[:3]],
            self.temp_dir,
            'test_model'
        )
        
        # Processar
        summary = processor.process_batch(job)
        
        # Verificar
        self.assertEqual(len(job.results), 2)
        self.assertEqual(len(job.errors), 1)
        self.assertEqual(summary['processed'], 2)
        self.assertEqual(summary['failed'], 1)

class TestImageComparison(unittest.TestCase):
    """Testes para comparação de imagens"""
    
    def test_compare_identical_images(self):
        """Testa comparação de imagens idênticas"""
        system = ImageComparisonSystem()
        
        # Criar imagem
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        # Comparar consigo mesma
        result = system.compare_images(image, image)
        
        # Verificar
        self.assertAlmostEqual(result.similarity_score, 1.0, places=2)
        self.assertAlmostEqual(result.structural_similarity, 1.0, places=2)
        self.assertEqual(len(result.regions_changed), 0)
    
    def test_compare_different_images(self):
        """Testa comparação de imagens diferentes"""
        system = ImageComparisonSystem()
        
        # Criar imagens diferentes
        image1 = np.zeros((256, 256), dtype=np.uint8)
        image2 = np.ones((256, 256), dtype=np.uint8) * 255
        
        # Comparar
        result = system.compare_images(image1, image2)
        
        # Verificar
        self.assertLess(result.similarity_score, 0.5)
        self.assertLess(result.structural_similarity, 0.5)
        self.assertGreater(len(result.regions_changed), 0)
    
    def test_detect_small_changes(self):
        """Testa detecção de mudanças pequenas"""
        system = ImageComparisonSystem()
        
        # Criar imagem base
        image1 = np.zeros((256, 256), dtype=np.uint8)
        image2 = image1.copy()
        
        # Adicionar pequena mudança
        cv2.circle(image2, (128, 128), 10, 255, -1)
        
        # Comparar
        result = system.compare_images(image1, image2)
        
        # Verificar
        self.assertGreater(result.similarity_score, 0.9)  # Ainda muito similar
        self.assertGreater(len(result.regions_changed), 0)  # Mas detecta mudança
        
        # Verificar região detectada
        region = result.regions_changed[0]
        self.assertIn('bbox', region)
        self.assertIn('area', region)
        self.assertGreater(region['area'], 0)

class TestSecurityAudit(unittest.TestCase):
    """Testes para segurança e auditoria"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / 'security_config.json'
        
        # Criar configuração temporária
        config = {
            'audit_db_path': str(Path(self.temp_dir) / 'audit.db'),
            'encryption_key_path': str(Path(self.temp_dir) / 'key.key'),
            'jwt_secret': 'test_secret'
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
        
        self.security_mgr = SecurityManager(str(self.config_path))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_create_user(self):
        """Testa criação de usuário"""
        success, result = self.security_mgr.create_user(
            username='test_user',
            password='Test@Password123',
            role=UserRole.RADIOLOGIST,
            email='test@example.com',
            full_name='Test User',
            created_by='system'
        )
        
        # Verificar
        self.assertTrue(success)
        self.assertIsInstance(result, str)  # user_id
    
    def test_password_validation(self):
        """Testa validação de senha"""
        # Senha fraca
        is_valid, msg = self.security_mgr._validate_password('123456')
        self.assertFalse(is_valid)
        
        # Senha sem caractere especial
        is_valid, msg = self.security_mgr._validate_password('TestPassword123')
        self.assertFalse(is_valid)
        
        # Senha válida
        is_valid, msg = self.security_mgr._validate_password('Test@Password123')
        self.assertTrue(is_valid)
    
    def test_authentication(self):
        """Testa autenticação"""
        # Criar usuário
        self.security_mgr.create_user(
            username='auth_test',
            password='Test@Password123',
            role=UserRole.PHYSICIAN,
            email='auth@test.com',
            full_name='Auth Test',
            created_by='system'
        )
        
        # Autenticar com senha correta
        token = self.security_mgr.authenticate(
            'auth_test',
            'Test@Password123',
            '127.0.0.1'
        )
        
        self.assertIsNotNone(token)
        
        # Autenticar com senha incorreta
        token = self.security_mgr.authenticate(
            'auth_test',
            'WrongPassword',
            '127.0.0.1'
        )
        
        self.assertIsNone(token)
    
    def test_audit_logging(self):
        """Testa log de auditoria"""
        # Registrar evento
        self.security_mgr.audit_event(
            AuditEventType.LOGIN,
            'test_user',
            '192.168.1.1',
            {'success': True},
            True,
            risk_level=0
        )
        
        # Recuperar logs
        logs = self.security_mgr.get_audit_logs()
        
        # Verificar
        self.assertGreater(len(logs), 0)
        self.assertEqual(logs[0]['event_type'], 'login')
        self.assertEqual(logs[0]['user_id'], 'test_user')
    
    def test_encryption(self):
        """Testa criptografia de dados"""
        # Dados de teste
        data = "Dados sensíveis do paciente".encode('utf-8')
        
        # Criptografar
        encrypted = self.security_mgr.encrypt_data(data)
        
        # Descriptografar
        decrypted = self.security_mgr.decrypt_data(encrypted)
        
        # Verificar
        self.assertNotEqual(encrypted, data)
        self.assertEqual(decrypted, data)

class TestExportSystem(unittest.TestCase):
    """Testes para sistema de exportação"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.export_mgr = ExportManager()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_export_model_h5(self):
        """Testa exportação de modelo H5"""
        # Criar modelo simples
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        
        # Exportar
        output_path = Path(self.temp_dir) / 'model.h5'
        info = self.export_mgr.export_model(
            model,
            str(output_path),
            format='h5'
        )
        
        # Verificar
        self.assertTrue(output_path.exists())
        self.assertEqual(info['format'], 'h5')
        self.assertGreater(info['size_bytes'], 0)
    
    def test_export_medical_image_dicom(self):
        """Testa exportação de imagem DICOM"""
        # Criar imagem
        image = np.random.randint(0, 4096, (256, 256), dtype=np.uint16)
        
        # Exportar
        output_path = Path(self.temp_dir) / 'test.dcm'
        info = self.export_mgr.export_medical_image(
            image,
            str(output_path),
            format='dcm',
            patient_info={'PatientName': 'Test', 'PatientID': '123'}
        )
        
        # Verificar
        self.assertTrue(output_path.exists())
        self.assertEqual(info['format'], 'dcm')
        
        # Verificar DICOM válido
        ds = pydicom.dcmread(str(output_path))
        self.assertEqual(ds.Rows, 256)
        self.assertEqual(ds.Columns, 256)
    
    def test_export_data_formats(self):
        """Testa exportação em diferentes formatos de dados"""
        # Criar DataFrame de teste
        df = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003'],
            'diagnosis': ['Normal', 'Pneumonia', 'Normal'],
            'confidence': [0.95, 0.87, 0.92]
        })
        
        formats = ['csv', 'json', 'xlsx', 'parquet']
        
        for fmt in formats:
            output_path = Path(self.temp_dir) / f'data.{fmt}'
            info = self.export_mgr.export_data(
                df,
                str(output_path),
                format=fmt
            )
            
            # Verificar
            self.assertTrue(output_path.exists())
            self.assertEqual(info['format'], fmt)
            self.assertEqual(info['num_rows'], 3)
            self.assertEqual(info['num_columns'], 3)
    
    def test_create_archive(self):
        """Testa criação de arquivo compactado"""
        # Criar arquivos de teste
        files = []
        for i in range(5):
            file_path = Path(self.temp_dir) / f'file_{i}.txt'
            file_path.write_text(f'Content {i}')
            files.append(str(file_path))
        
        # Criar arquivo ZIP
        archive_path = Path(self.temp_dir) / 'archive.zip'
        info = self.export_mgr.create_archive(
            files,
            str(archive_path),
            format='zip'
        )
        
        # Verificar
        self.assertTrue(archive_path.exists())
        self.assertEqual(info['num_files'], 5)
        self.assertGreater(info['compression_ratio'], 0)

# Testes de Integração
class TestIntegrationPipeline(unittest.TestCase):
    """Testes de integração do pipeline completo"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.integration
    def test_full_analysis_pipeline(self):
        """Testa pipeline completo de análise"""
        # 1. Criar imagem DICOM
        ds = sample_dicom()
        dicom_path = Path(self.temp_dir) / 'test.dcm'
        ds.save_as(str(dicom_path))
        
        # 2. Processar DICOM
        processor = DICOMProcessor()
        loaded_ds = processor.read_dicom(dicom_path)
        image_array = processor.dicom_to_array(loaded_ds)
        metadata = processor.extract_metadata(loaded_ds)
        
        # 3. Pré-processar para IA
        processed = processor.preprocess_for_ai(
            image_array,
            target_size=(224, 224)
        )
        
        # 4. Simular inferência (mock)
        result = PredictionResult(
            image_path=str(dicom_path),
            predictions={'Normal': 0.85, 'Abnormal': 0.15},
            predicted_class='Normal',
            confidence=0.85,
            processing_time=0.5,
            metadata=metadata
        )
        
        # 5. Exportar resultados
        export_mgr = ExportManager()
        
        # Exportar para JSON
        json_path = Path(self.temp_dir) / 'results.json'
        export_mgr.export_data(
            result.__dict__,
            str(json_path),
            format='json'
        )
        
        # Verificar
        self.assertTrue(json_path.exists())
        
        with open(json_path, 'r') as f:
            saved_results = json.load(f)
        
        self.assertEqual(saved_results['predicted_class'], 'Normal')
        self.assertEqual(saved_results['confidence'], 0.85)
    
    @pytest.mark.integration
    def test_batch_processing_pipeline(self):
        """Testa pipeline de processamento em lote"""
        # Criar múltiplas imagens
        image_files = []
        for i in range(5):
            img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            img_path = Path(self.temp_dir) / f'image_{i}.png'
            cv2.imwrite(str(img_path), img)
            image_files.append(str(img_path))
        
        # Mock do engine
        mock_engine = MagicMock()
        mock_engine.predict_single.return_value = PredictionResult(
            image_path='',
            predictions={'Normal': 0.9, 'Abnormal': 0.1},
            predicted_class='Normal',
            confidence=0.9,
            processing_time=0.1
        )
        
        # Processar em lote
        processor = BatchProcessor(mock_engine, max_workers=2)
        job = processor.create_batch_job(
            image_files,
            self.temp_dir,
            'test_model'
        )
        
        summary = processor.process_batch(job)
        
        # Verificar resultados
        results_csv = Path(self.temp_dir) / f'{job.job_id}_results.csv'
        self.assertTrue(results_csv.exists())
        
        # Carregar e verificar CSV
        df = pd.read_csv(results_csv)
        self.assertEqual(len(df), 5)
        self.assertIn('predicted_class', df.columns)
        self.assertIn('confidence', df.columns)

# Testes de Performance
class TestPerformance(unittest.TestCase):
    """Testes de performance do sistema"""
    
    @pytest.mark.performance
    def test_inference_speed(self):
        """Testa velocidade de inferência"""
        # Criar modelo simples
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, input_shape=(224, 224, 1)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        # Criar batch de imagens
        batch_sizes = [1, 8, 16, 32]
        
        for batch_size in batch_sizes:
            batch = np.random.random((batch_size, 224, 224, 1)).astype(np.float32)
            
            # Medir tempo
            start_time = time.time()
            predictions = model.predict(batch, verbose=0)
            inference_time = time.time() - start_time
            
            # Calcular FPS
            fps = batch_size / inference_time
            
            logger.info(f"Batch size: {batch_size}, Time: {inference_time:.3f}s, FPS: {fps:.1f}")
            
            # Verificar performance mínima
            self.assertGreater(fps, 10)  # Mínimo 10 FPS
    
    @pytest.mark.performance
    def test_memory_usage(self):
        """Testa uso de memória"""
        # Monitorar memória inicial
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Criar e processar imagens grandes
        for _ in range(10):
            image = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
            
            # Simular processamento
            processed = cv2.resize(image, (224, 224))
            _ = processed.mean()
        
        # Verificar memória final
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        logger.info(f"Memory increase: {memory_increase:.1f} MB")
        
        # Verificar que não há vazamento significativo
        self.assertLess(memory_increase, 100)  # Menos de 100MB de aumento
    
    @pytest.mark.performance
    def test_batch_processing_scalability(self):
        """Testa escalabilidade do processamento em lote"""
        # Criar arquivos de teste
        num_files = 100
        files = []
        
        temp_dir = tempfile.mkdtemp()
        try:
            for i in range(num_files):
                img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
                img_path = Path(temp_dir) / f'img_{i}.png'
                cv2.imwrite(str(img_path), img)
                files.append(str(img_path))
            
            # Mock do engine
            mock_engine = MagicMock()
            mock_engine.predict_single.return_value = PredictionResult(
                image_path='',
                predictions={'Normal': 0.9, 'Abnormal': 0.1},
                predicted_class='Normal',
                confidence=0.9,
                processing_time=0.01  # Simular processamento rápido
            )
            
            # Testar com diferentes números de workers
            worker_counts = [1, 2, 4, 8]
            
            for workers in worker_counts:
                processor = BatchProcessor(mock_engine, max_workers=workers)
                job = processor.create_batch_job(files, temp_dir, 'test')
                
                start_time = time.time()
                processor.process_batch(job)
                elapsed_time = time.time() - start_time
                
                throughput = num_files / elapsed_time
                logger.info(f"Workers: {workers}, Time: {elapsed_time:.2f}s, "
                           f"Throughput: {throughput:.1f} files/s")
                
                # Verificar melhoria com paralelização
                if workers > 1:
                    self.assertGreater(throughput, num_files / elapsed_time * 0.8)
        
        finally:
            shutil.rmtree(temp_dir)

# Testes de Stress
class TestStress(unittest.TestCase):
    """Testes de stress do sistema"""
    
    @pytest.mark.stress
    def test_concurrent_users(self):
        """Testa múltiplos usuários concorrentes"""
        security_mgr = SecurityManager()
        
        # Criar múltiplos usuários
        users = []
        for i in range(10):
            success, user_id = security_mgr.create_user(
                username=f'user_{i}',
                password=f'Pass@{i}123',
                role=UserRole.VIEWER,
                email=f'user{i}@test.com',
                full_name=f'User {i}',
                created_by='system'
            )
            if success:
                users.append((f'user_{i}', f'Pass@{i}123'))
        
        # Simular logins concorrentes
        import concurrent.futures
        
        def login_user(credentials):
            username, password = credentials
            return security_mgr.authenticate(username, password, '127.0.0.1')
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(login_user, user) for user in users]
            tokens = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # Verificar que todos conseguiram fazer login
        valid_tokens = [t for t in tokens if t is not None]
        self.assertEqual(len(valid_tokens), len(users))
    
    @pytest.mark.stress
    def test_large_image_processing(self):
        """Testa processamento de imagens muito grandes"""
        # Criar imagem grande (4K)
        large_image = np.random.randint(0, 255, (3840, 2160), dtype=np.uint8)
        
        processor = DICOMProcessor()
        
        # Processar
        start_time = time.time()
        processed = processor.preprocess_for_ai(
            large_image,
            target_size=(224, 224)
        )
        processing_time = time.time() - start_time
        
        # Verificar
        self.assertEqual(processed.shape, (224, 224, 1))
        self.assertLess(processing_time, 2.0)  # Deve processar em menos de 2 segundos

# Utilitários de teste
def create_test_dataset(num_samples: int, 
                       num_classes: int,
                       image_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """Cria dataset sintético para testes"""
    X = []
    y = []
    
    for i in range(num_samples):
        # Criar imagem com padrão baseado na classe
        class_idx = i % num_classes
        image = np.zeros(image_size, dtype=np.uint8)
        
        # Adicionar padrão específico da classe
        if class_idx == 0:
            cv2.circle(image, (112, 112), 50, 255, -1)
        elif class_idx == 1:
            cv2.rectangle(image, (50, 50), (174, 174), 255, -1)
        else:
            cv2.ellipse(image, (112, 112), (80, 40), 0, 0, 360, 255, -1)
        
        # Adicionar ruído
        noise = np.random.normal(0, 20, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        X.append(image)
        y.append(class_idx)
    
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes)
    
    return X, y

def run_all_tests():
    """Executa todos os testes"""
    # Configurar pytest
    pytest_args = [
        '-v',  # Verbose
        '--tb=short',  # Traceback curto
        '--color=yes',  # Saída colorida
        '-m', 'not slow and not integration and not performance and not stress',  # Apenas testes rápidos
        __file__
    ]
    
    # Executar testes
    return pytest.main(pytest_args)

if __name__ == '__main__':
    run_all_tests()
