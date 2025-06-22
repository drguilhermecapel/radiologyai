import os
import sys
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class NIHXrayTrainerAdaptive:
    def __init__(self, data_path='D:/NIH_CHEST_XRAY/', batch_size=16):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.image_size = (224, 224)
        self.num_classes = 14
        self.start_time = datetime.now()
        self.images_dir = None  # Sera detectado automaticamente
        
        # Patologias
        self.pathologies = [
            'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
            'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
            'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
        ]
        
        self.setup_logging()
        self.configure_tensorflow()
        
    def setup_logging(self):
        """Configura logging"""
        os.makedirs('logs', exist_ok=True)
        log_file = f'logs/nih_training_{self.start_time.strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('NIH-Xray')
        self.logger.info("Sistema de Treinamento NIH Chest X-ray Iniciado")
        
    def configure_tensorflow(self):
        """Configura TensorFlow"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self.logger.info(f"GPU encontrada: {gpus[0].name}")
        else:
            self.logger.info("Usando CPU")
            
    def find_images_directory(self):
        """Procura automaticamente onde as imagens estao"""
        self.logger.info("Procurando diretorio de imagens...")
        
        # Possiveis nomes de diretorios
        possible_dirs = ['images', 'Images', 'IMAGE', 'image', 'imgs', 'Imgs']
        
        # Procurar no diretorio base
        for dir_name in possible_dirs:
            test_path = self.data_path / dir_name
            if test_path.exists() and test_path.is_dir():
                self.images_dir = test_path
                self.logger.info(f"Diretorio de imagens encontrado: {self.images_dir}")
                return True
                
        # Se nao encontrou, procurar por arquivos PNG
        png_files = list(self.data_path.rglob('*.png'))
        if png_files:
            # Pegar o diretorio do primeiro arquivo
            self.images_dir = png_files[0].parent
            self.logger.info(f"Imagens encontradas em: {self.images_dir}")
            self.logger.info(f"Total de imagens PNG: {len(png_files)}")
            return True
            
        self.logger.error("Nenhum diretorio de imagens encontrado!")
        return False
        
    def load_metadata(self):
        """Carrega metadados"""
        csv_path = self.data_path / 'Data_Entry_2017_v2020.csv'
        
        if not csv_path.exists():
            # Tentar variações do nome
            alt_names = ['Data_Entry_2017.csv', 'Data_Entry.csv']
            for name in alt_names:
                alt_path = self.data_path / name
                if alt_path.exists():
                    csv_path = alt_path
                    break
                    
        self.logger.info(f"Carregando metadados de {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # Encontrar diretorio de imagens
        if not self.find_images_directory():
            return False
            
        # Verificar se consegue encontrar as imagens
        sample_image = self.df['Image Index'].iloc[0]
        test_paths = [
            self.images_dir / sample_image,
            self.data_path / sample_image,
            self.images_dir / f"{sample_image}.png" if not sample_image.endswith('.png') else self.images_dir / sample_image
        ]
        
        image_found = False
        for test_path in test_paths:
            if test_path.exists():
                self.logger.info(f"Formato de caminho correto: {test_path}")
                image_found = True
                break
                
        if not image_found:
            self.logger.error(f"Nao foi possivel encontrar a imagem de teste: {sample_image}")
            self.logger.error("Verifique a estrutura do dataset")
            return False
            
        # Processar labels
        self.process_labels()
        
        # Usar subset para teste
        self.logger.info("Usando subset de 5000 imagens para teste rapido")
        self.df = self.df.sample(n=min(5000, len(self.df)), random_state=42)
        
        self.logger.info(f"Total de imagens: {len(self.df)}")
        return True
        
    def process_labels(self):
        """Processa labels"""
        for pathology in self.pathologies:
            self.df[pathology] = self.df['Finding Labels'].apply(
                lambda x: 1 if pathology in x else 0
            )
            
        self.logger.info("\nDistribuicao de patologias:")
        for pathology in self.pathologies:
            count = self.df[pathology].sum()
            percentage = (count / len(self.df)) * 100
            self.logger.info(f"  {pathology}: {count} ({percentage:.1f}%)")
            
    def create_data_splits(self):
        """Cria splits de dados"""
        self.logger.info("\nCriando divisoes de dados...")
        
        patient_ids = self.df['Patient ID'].unique()
        
        # Splits
        patients_temp, patients_test = train_test_split(
            patient_ids, test_size=0.15, random_state=42
        )
        
        patients_train, patients_val = train_test_split(
            patients_temp, test_size=0.15/0.85, random_state=42
        )
        
        self.train_df = self.df[self.df['Patient ID'].isin(patients_train)].reset_index(drop=True)
        self.val_df = self.df[self.df['Patient ID'].isin(patients_val)].reset_index(drop=True)
        self.test_df = self.df[self.df['Patient ID'].isin(patients_test)].reset_index(drop=True)
        
        self.logger.info(f"Treino: {len(self.train_df)} imagens")
        self.logger.info(f"Validacao: {len(self.val_df)} imagens")
        self.logger.info(f"Teste: {len(self.test_df)} imagens")
        
    def create_data_generators(self):
        """Cria geradores de dados"""
        self.logger.info("\nCriando geradores de dados...")
        
        # Preparar dataframes
        def prepare_dataframe(df):
            df = df.copy()
            df['filename'] = df['Image Index']
            return df
            
        train_df_gen = prepare_dataframe(self.train_df)
        val_df_gen = prepare_dataframe(self.val_df)
        
        # Augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Geradores
        try:
            self.train_generator = train_datagen.flow_from_dataframe(
                train_df_gen,
                directory=str(self.images_dir),
                x_col='filename',
                y_col=self.pathologies,
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode='raw',
                shuffle=True
            )
            
            self.val_generator = val_datagen.flow_from_dataframe(
                val_df_gen,
                directory=str(self.images_dir),
                x_col='filename',
                y_col=self.pathologies,
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode='raw',
                shuffle=False
            )
            
            self.logger.info("Geradores criados com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao criar geradores: {str(e)}")
            raise
            
    def create_simple_model(self):
        """Cria modelo mais simples para economizar memoria"""
        self.logger.info("\nCriando modelo CNN simples...")
        
        model = models.Sequential([
            # Entrada
            layers.Input(shape=(*self.image_size, 3)),
            
            # Blocos convolucionais
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Camadas densas
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # Saida
            layers.Dense(self.num_classes, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'binary_accuracy',
                tf.keras.metrics.AUC(name='auc', multi_label=True)
            ]
        )
        
        self.model = model
        self.logger.info(f"Modelo criado com {model.count_params():,} parametros")
        
    def train(self, epochs=10):
        """Treina modelo"""
        self.logger.info("\nIniciando treinamento...")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=3,
                mode='max',
                restore_best_weights=True
            ),
            
            tf.keras.callbacks.ModelCheckpoint(
                'best_nih_simple_model.h5',
                monitor='val_auc',
                save_best_only=True,
                mode='max'
            )
        ]
        
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
        
    def run(self):
        """Executa pipeline"""
        try:
            # Carregar dados
            if not self.load_metadata():
                return
                
            # Splits
            self.create_data_splits()
            
            # Geradores
            self.create_data_generators()
            
            # Modelo
            self.create_simple_model()
            
            # Treinar
            self.train(epochs=5)
            
            self.logger.info("\nTreinamento concluido!")
            self.logger.info(f"Modelo salvo: best_nih_simple_model.h5")
            
        except Exception as e:
            self.logger.error(f"Erro: {str(e)}", exc_info=True)

if __name__ == "__main__":
    print("Sistema Adaptativo de Treinamento NIH Chest X-ray")
    print("="*50)
    
    trainer = NIHXrayTrainerAdaptive(
        data_path='D:/NIH_CHEST_XRAY/',
        batch_size=16
    )
    trainer.run()