"""
MedAI Setup & Initialize - Script de Configuração Inicial
Cria estrutura de diretórios, usuários padrão e modelos de exemplo
"""

import os
import sys
import json
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from datetime import datetime
import sqlite3
import bcrypt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import pydicom
from pydicom.dataset import Dataset, FileDataset

# Console para output formatado
console = Console()

class MedAISetup:
    """Classe para configuração inicial do MedAI"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / 'data'
        self.models_dir = self.base_dir / 'models'
        self.logs_dir = self.base_dir / 'logs'
        self.reports_dir = self.base_dir / 'reports'
        self.db_path = self.base_dir / 'medai_security.db'
        
    def run(self):
        """Executa todo o processo de setup"""
        console.print("[bold green]🚀 MedAI - Configuração Inicial[/bold green]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            # 1. Criar estrutura de diretórios
            task1 = progress.add_task("Criando estrutura de diretórios...", total=4)
            self.create_directory_structure()
            progress.update(task1, completed=4)
            
            # 2. Inicializar banco de dados
            task2 = progress.add_task("Inicializando banco de dados...", total=1)
            self.initialize_database()
            progress.update(task2, completed=1)
            
            # 3. Criar usuários padrão
            task3 = progress.add_task("Criando usuários padrão...", total=3)
            self.create_default_users()
            progress.update(task3, completed=3)
            
            # 4. Gerar imagens DICOM de exemplo
            task4 = progress.add_task("Gerando imagens DICOM de exemplo...", total=10)
            self.generate_sample_dicom_images(progress, task4)
            
            # 5. Criar modelos de exemplo
            task5 = progress.add_task("Criando modelos de exemplo...", total=5)
            self.create_sample_models(progress, task5)
            
            # 6. Criar arquivo de configuração
            task6 = progress.add_task("Criando arquivo de configuração...", total=1)
            self.create_config_file()
            progress.update(task6, completed=1)
        
        console.print("\n✅ [bold green]Configuração concluída com sucesso![/bold green]")
        self.print_summary()
    
    def create_directory_structure(self):
        """Cria estrutura de diretórios necessária"""
        directories = [
            self.data_dir,
            self.data_dir / 'samples',
            self.data_dir / 'samples' / 'normal',
            self.data_dir / 'samples' / 'pneumonia',
            self.data_dir / 'pacs_downloads',
            self.models_dir,
            self.logs_dir,
            self.reports_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def initialize_database(self):
        """Inicializa banco de dados de segurança"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Criar tabelas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                name TEXT,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_default_users(self):
        """Cria usuários padrão para demonstração"""
        users = [
            {
                'username': 'admin',
                'password': 'admin123',
                'role': 'admin',
                'name': 'Administrador',
                'email': 'admin@medai.local'
            },
            {
                'username': 'radiologist',
                'password': 'rad123',
                'role': 'radiologist',
                'name': 'Dr. Silva',
                'email': 'radiologist@medai.local'
            },
            {
                'username': 'viewer',
                'password': 'view123',
                'role': 'viewer',
                'name': 'Visualizador',
                'email': 'viewer@medai.local'
            }
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for user in users:
            # Hash da senha
            password_hash = bcrypt.hashpw(
                user['password'].encode('utf-8'),
                bcrypt.gensalt()
            ).decode('utf-8')
            
            try:
                cursor.execute('''
                    INSERT INTO users (username, password_hash, role, name, email)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user['username'], password_hash, user['role'], user['name'], user['email']))
            except sqlite3.IntegrityError:
                # Usuário já existe
                pass
        
        conn.commit()
        conn.close()
    
    def generate_sample_dicom_images(self, progress, task):
        """Gera imagens DICOM sintéticas para testes"""
        # Parâmetros para imagens sintéticas
        image_size = (512, 512)
        
        # Gerar imagens normais
        for i in range(5):
            # Criar imagem sintética (pulmão normal)
            image = self._generate_synthetic_chest_xray('normal')
            
            # Criar dataset DICOM
            ds = self._create_dicom_dataset(
                image,
                patient_id=f"NORMAL{i+1:03d}",
                patient_name=f"Paciente Normal {i+1}",
                study_description="Radiografia de Tórax - Normal"
            )
            
            # Salvar arquivo
            filename = self.data_dir / 'samples' / 'normal' / f'normal_{i+1:03d}.dcm'
            ds.save_as(filename, write_like_original=False)
            
            progress.update(task, advance=1)
        
        # Gerar imagens com pneumonia
        for i in range(5):
            # Criar imagem sintética (pneumonia)
            image = self._generate_synthetic_chest_xray('pneumonia')
            
            # Criar dataset DICOM
            ds = self._create_dicom_dataset(
                image,
                patient_id=f"PNEUM{i+1:03d}",
                patient_name=f"Paciente Pneumonia {i+1}",
                study_description="Radiografia de Tórax - Pneumonia"
            )
            
            # Salvar arquivo
            filename = self.data_dir / 'samples' / 'pneumonia' / f'pneumonia_{i+1:03d}.dcm'
            ds.save_as(filename, write_like_original=False)
            
            progress.update(task, advance=1)
    
    def _generate_synthetic_chest_xray(self, condition='normal'):
        """
        Gera uma imagem sintética de radiografia de tórax
        
        Esta é uma representação simplificada apenas para demonstração.
        Em um sistema real, seriam usadas imagens médicas reais.
        """
        # Criar imagem base
        image = np.zeros((512, 512), dtype=np.uint16)
        
        # Adicionar gradiente de fundo (simulando tecido)
        x = np.linspace(-1, 1, 512)
        y = np.linspace(-1, 1, 512)
        X, Y = np.meshgrid(x, y)
        
        # Forma elíptica para simular tórax
        thorax = np.exp(-(X**2 / 0.6 + Y**2 / 0.8)) * 2000
        image += thorax.astype(np.uint16)
        
        # Adicionar "pulmões" (áreas mais escuras)
        lung_left = np.exp(-((X+0.3)**2 / 0.2 + (Y)**2 / 0.4)) * 1500
        lung_right = np.exp(-((X-0.3)**2 / 0.2 + (Y)**2 / 0.4)) * 1500
        image -= (lung_left + lung_right).astype(np.uint16)
        
        # Adicionar "coluna vertebral" (linha central mais clara)
        spine = np.exp(-(X**2 / 0.02 + Y**2 / 1)) * 800
        image += spine.astype(np.uint16)
        
        # Adicionar "costelas" (linhas horizontais)
        for i in range(-4, 5):
            rib_y = i * 0.15
            rib = np.exp(-(X**2 / 0.5 + (Y - rib_y)**2 / 0.01)) * 300
            image += rib.astype(np.uint16)
        
        # Se pneumonia, adicionar opacidades
        if condition == 'pneumonia':
            # Adicionar consolidações (áreas mais brancas nos pulmões)
            num_opacities = np.random.randint(2, 5)
            for _ in range(num_opacities):
                cx = np.random.uniform(-0.4, 0.4)
                cy = np.random.uniform(-0.3, 0.3)
                size = np.random.uniform(0.05, 0.15)
                opacity = np.exp(-((X-cx)**2 + (Y-cy)**2) / size**2) * 1000
                image += opacity.astype(np.uint16)
        
        # Adicionar ruído realista
        noise = np.random.normal(0, 50, image.shape)
        image = np.clip(image + noise, 0, 4095).astype(np.uint16)
        
        return image
    
    def _create_dicom_dataset(self, pixel_array, patient_id, patient_name, study_description):
        """Cria um dataset DICOM com metadados apropriados"""
        # Criar o dataset
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.DigitalXRayImageStorageForPresentation
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
        
        # Criar o dataset principal
        ds = FileDataset(
            None,
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128
        )
        
        # Adicionar metadados do paciente
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.PatientBirthDate = "19800101"
        ds.PatientSex = "M"
        
        # Adicionar metadados do estudo
        ds.StudyDate = datetime.now().strftime("%Y%m%d")
        ds.StudyTime = datetime.now().strftime("%H%M%S")
        ds.StudyDescription = study_description
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        
        # Adicionar metadados da série
        ds.SeriesDate = ds.StudyDate
        ds.SeriesTime = ds.StudyTime
        ds.SeriesDescription = "PA"
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesNumber = 1
        
        # Adicionar metadados da imagem
        ds.InstanceNumber = 1
        ds.ImageType = ["ORIGINAL", "PRIMARY"]
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "DX"  # Digital Radiography
        
        # Adicionar dados de pixel
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = pixel_array.shape[0]
        ds.Columns = pixel_array.shape[1]
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.PixelData = pixel_array.tobytes()
        
        # Adicionar informações de janelamento
        ds.WindowCenter = 2048
        ds.WindowWidth = 4096
        
        # Adicionar informações de equipamento
        ds.Manufacturer = "MedAI Simulator"
        ds.InstitutionName = "MedAI Hospital"
        ds.ManufacturerModelName = "MedAI X-Ray v1.0"
        
        return ds
    
    def create_sample_models(self, progress, task):
        """Cria modelos de exemplo pré-treinados"""
        # Para demonstração, criamos modelos pequenos com pesos aleatórios
        # Em produção, estes seriam modelos realmente treinados
        
        models_to_create = [
            ('densenet', self._create_densenet_model),
            ('resnet', self._create_resnet_model),
            ('efficientnet', self._create_efficientnet_model),
            ('custom_cnn', self._create_custom_cnn_model),
            ('attention_unet', self._create_attention_unet_model)
        ]
        
        for model_name, create_func in models_to_create:
            try:
                # Criar modelo
                model = create_func()
                
                # Salvar pesos (inicializados aleatoriamente para demo)
                weights_path = self.models_dir / f"{model_name}_weights.h5"
                model.save_weights(str(weights_path))
                
                # Salvar modelo completo também
                model_path = self.models_dir / f"{model_name}_model.h5"
                model.save(str(model_path))
                
            except Exception as e:
                console.print(f"⚠️  Aviso: Não foi possível criar modelo {model_name}: {e}")
            
            progress.update(task, advance=1)
    
    def _create_densenet_model(self):
        """Cria modelo DenseNet simplificado"""
        base_model = tf.keras.applications.DenseNet121(
            weights=None,  # Sem pesos pré-treinados para demo
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(5, activation='softmax')  # 5 classes
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_resnet_model(self):
        """Cria modelo ResNet simplificado"""
        base_model = tf.keras.applications.ResNet50(
            weights=None,
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_efficientnet_model(self):
        """Cria modelo EfficientNet simplificado"""
        # Modelo customizado simples (EfficientNet requer instalação adicional)
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_custom_cnn_model(self):
        """Cria modelo CNN customizado"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_attention_unet_model(self):
        """Cria modelo U-Net com atenção simplificado"""
        # Para segmentação - modelo simplificado usando Sequential para evitar KerasTensor
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', input_shape=(256, 256, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='same')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_config_file(self):
        """Cria arquivo de configuração JSON"""
        config = {
            "version": "1.0.0",
            "paths": {
                "data_dir": str(self.data_dir),
                "models_dir": str(self.models_dir),
                "logs_dir": str(self.logs_dir),
                "reports_dir": str(self.reports_dir)
            },
            "models": {
                "default": "densenet",
                "available": ["densenet", "resnet", "efficientnet", "custom_cnn", "attention_unet"]
            },
            "pacs": {
                "ae_title": "MEDAI_SCU",
                "host": "localhost",
                "port": 11112,
                "timeout": 30
            },
            "security": {
                "jwt_secret": "your-secret-key-here-change-in-production",
                "jwt_expiration_hours": 24,
                "encryption_key": "your-encryption-key-here-change-in-production"
            }
        }
        
        config_path = self.base_dir / 'medai_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def print_summary(self):
        """Imprime resumo da configuração"""
        console.print("\n📋 [bold]Resumo da Configuração:[/bold]")
        console.print(f"  • Diretório base: {self.base_dir}")
        console.print(f"  • Banco de dados: {self.db_path}")
        console.print(f"  • Imagens de exemplo: {self.data_dir / 'samples'}")
        console.print(f"  • Modelos salvos em: {self.models_dir}")
        
        console.print("\n👥 [bold]Usuários criados:[/bold]")
        console.print("  • admin / admin123 (Administrador)")
        console.print("  • radiologist / rad123 (Radiologista)")
        console.print("  • viewer / view123 (Visualizador)")
        
        console.print("\n🚀 [bold]Próximos passos:[/bold]")
        console.print("  1. Execute a GUI: python medai_gui_complete.py")
        console.print("  2. Ou use a CLI: python medai_cli_complete.py --help")
        console.print("  3. Faça login com um dos usuários criados")
        console.print("  4. Teste com as imagens em data/samples/")
        
        console.print("\n⚠️  [bold yellow]Importante:[/bold yellow]")
        console.print("  • As imagens DICOM geradas são sintéticas (apenas para demonstração)")
        console.print("  • Os modelos têm pesos aleatórios (precisam ser treinados com dados reais)")
        console.print("  • Para produção, use imagens médicas reais e modelos treinados adequadamente")

class SystemInitializer:
    """Classe para inicialização do sistema MedAI"""
    
    def __init__(self):
        self.setup = MedAISetup()
    
    def initialize_system(self):
        """Inicializa o sistema se necessário"""
        try:
            if not self.setup.db_path.exists():
                console.print("🔧 Configuração inicial necessária...")
                self.setup.run()
                return True
            else:
                console.print("✅ Sistema já configurado")
                return True
        except Exception as e:
            console.print(f"❌ Erro na inicialização: {e}")
            return False

def main():
    """Função principal"""
    setup = MedAISetup()
    
    console.print("[bold]🏥 MedAI - Sistema de Análise de Imagens Médicas[/bold]")
    console.print("Este script irá configurar o ambiente inicial do MedAI.\n")
    
    # Verificar se já existe configuração
    if setup.db_path.exists():
        console.print("⚠️  [yellow]Configuração existente detectada.[/yellow]")
        response = input("Deseja reconfigurar? Isso pode sobrescrever dados existentes. (s/N): ")
        if response.lower() != 's':
            console.print("Configuração cancelada.")
            return
    
    # Executar setup
    try:
        setup.run()
    except Exception as e:
        console.print(f"\n❌ [bold red]Erro durante configuração:[/bold red] {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
