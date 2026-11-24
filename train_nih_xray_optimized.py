import os
import sys
import gc
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class NIHXrayTrainer:
    def __init__(self, data_path='D:/NIH_CHEST_XRAY/', batch_size=16):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.image_size = (256, 256)  # Reduzido para economizar RAM
        self.num_classes = 14
        self.start_time = datetime.now()
        
        # Patologias do NIH dataset
        self.pathologies = [
            'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
            'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
            'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
        ]
        
        self.setup_logging()
        self.configure_tensorflow()
        
    def setup_logging(self):
        """Configura sistema de logging"""
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
        self.logger.info(f"TensorFlow versao: {tf.__version__}")
        
    def configure_tensorflow(self):
        """Configura TensorFlow para uso eficiente de memoria"""
        # Limitar crescimento de memoria
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self.logger.info(f"GPU encontrada: {gpus[0].name}")
        else:
            self.logger.info("Usando CPU - treinamento sera mais lento")
            
        # Configurar para usar menos memoria
        tf.config.threading.set_inter_op_parallelism_threads(4)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        
    def load_metadata(self):
        """Carrega e processa metadados do dataset"""
        csv_path = self.data_path / 'Data_Entry_2017_v2020.csv'
        
        self.logger.info(f"Carregando metadados de {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # Adicionar caminho completo das imagens
        self.df['path'] = self.df['Image Index'].apply(
            lambda x: str(self.data_path / 'images' / x)
        )
        
        # Verificar se as imagens existem (apenas uma amostra)
        sample_path = self.df['path'].iloc[0]
        if not os.path.exists(sample_path):
            self.logger.error(f"Imagem nao encontrada: {sample_path}")
            self.logger.error("Verifique se as imagens estao na pasta 'images'")
            return False
            
        # Processar labels
        self.process_labels()
        
        # Estatisticas
        self.logger.info(f"Total de imagens: {len(self.df)}")
        self.logger.info(f"Pacientes unicos: {self.df['Patient ID'].nunique()}")
        
        return True
        
    def process_labels(self):
        """Processa labels multi-label"""
        # Criar colunas binarias para cada patologia
        for pathology in self.pathologies:
            self.df[pathology] = self.df['Finding Labels'].apply(
                lambda x: 1 if pathology in x else 0
            )
            
        # Adicionar coluna para "No Finding"
        self.df['No Finding'] = self.df['Finding Labels'].apply(
            lambda x: 1 if 'No Finding' in x else 0
        )
        
        # Estatisticas de patologias
        self.logger.info("\nDistribuicao de patologias:")
        for pathology in self.pathologies:
            count = self.df[pathology].sum()
            percentage = (count / len(self.df)) * 100
            self.logger.info(f"  {pathology}: {count} ({percentage:.1f}%)")
            
    def create_data_splits(self, test_size=0.15, val_size=0.15):
        """Cria divisoes treino/validacao/teste por paciente"""
        self.logger.info("\nCriando divisoes de dados...")
        
        # Dividir por paciente para evitar data leakage
        patient_ids = self.df['Patient ID'].unique()
        
        # Primeira divisao: treino+val vs teste
        patients_temp, patients_test = train_test_split(
            patient_ids, test_size=test_size, random_state=42
        )
        
        # Segunda divisao: treino vs validacao
        patients_train, patients_val = train_test_split(
            patients_temp, test_size=val_size/(1-test_size), random_state=42
        )
        
        # Criar dataframes
        self.train_df = self.df[self.df['Patient ID'].isin(patients_train)]
        self.val_df = self.df[self.df['Patient ID'].isin(patients_val)]
        self.test_df = self.df[self.df['Patient ID'].isin(patients_test)]
        
        # Balancear dataset de treino
        self.balance_training_data()
        
        self.logger.info(f"Treino: {len(self.train_df)} imagens")
        self.logger.info(f"Validacao: {len(self.val_df)} imagens")
        self.logger.info(f"Teste: {len(self.test_df)} imagens")
        
    def balance_training_data(self):
        """Balanceia dataset de treino usando undersampling"""
        # Encontrar casos positivos para cada patologia
        positive_cases = []
        for pathology in self.pathologies:
            positive_df = self.train_df[self.train_df[pathology] == 1]
            if len(positive_df) > 0:
                positive_cases.append(positive_df)
                
        # Casos normais
        normal_df = self.train_df[self.train_df['No Finding'] == 1]
        
        # Limitar casos normais para balancear
        max_positive = max(len(df) for df in positive_cases)
        normal_balanced = normal_df.sample(n=min(len(normal_df), max_positive * 2), random_state=42)
        
        # Combinar todos
        balanced_dfs = positive_cases + [normal_balanced]
        self.train_df = pd.concat(balanced_dfs).drop_duplicates()
        
        self.logger.info(f"Dataset balanceado: {len(self.train_df)} imagens")
        
    def create_generators(self):
        """Cria geradores de dados com augmentation"""
        # Augmentation para treino
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='constant',
            cval=0
        )
        
        # Sem augmentation para validacao/teste
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Funcao para carregar e preprocessar imagens
        def load_and_preprocess_image(path):
            # Carregar como grayscale
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
                
            # Resize
            img = cv2.resize(img, self.image_size)
            
            # CLAHE para melhorar contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
            # Converter para 3 canais (RGB)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            return img
            
        # Gerador customizado
        def custom_generator(df, datagen, batch_size):
            n = len(df)
            indices = np.arange(n)
            
            while True:
                # Embaralhar indices
                np.random.shuffle(indices)
                
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    batch_indices = indices[start:end]
                    
                    batch_images = []
                    batch_labels = []
                    
                    for idx in batch_indices:
                        row = df.iloc[idx]
                        
                        # Carregar imagem
                        img = load_and_preprocess_image(row['path'])
                        if img is None:
                            continue
                            
                        # Aplicar augmentation
                        img = datagen.random_transform(img)
                        img = datagen.standardize(img)
                        
                        batch_images.append(img)
                        batch_labels.append([row[p] for p in self.pathologies])
                        
                    if batch_images:
                        yield np.array(batch_images), np.array(batch_labels)
                        
        # Criar geradores
        self.train_generator = custom_generator(self.train_df, train_datagen, self.batch_size)
        self.val_generator = custom_generator(self.val_df, val_datagen, self.batch_size)
        
        self.steps_per_epoch = len(self.train_df) // self.batch_size
        self.validation_steps = len(self.val_df) // self.batch_size
        
    def create_model(self):
        """Cria modelo otimizado para chest X-ray"""
        self.logger.info("\nCriando modelo DenseNet121...")
        
        # Base model - DenseNet121 pre-treinada no ImageNet
        base_model = tf.keras.applications.DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.image_size, 3),
            pooling='avg'
        )
        
        # Fine-tuning: descongelar ultimas camadas
        for layer in base_model.layers[:-30]:
            layer.trainable = False
            
        # Modelo customizado
        inputs = tf.keras.Input(shape=(*self.image_size, 3))
        
        # Pre-processamento
        x = tf.keras.layers.Lambda(lambda x: tf.keras.applications.densenet.preprocess_input(x))(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Camadas adicionais com regularizacao
        x = tf.keras.layers.Dense(512, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        x = tf.keras.layers.Dense(256, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output layer com sigmoid para multi-label
        outputs = tf.keras.layers.Dense(self.num_classes, activation='sigmoid', name='predictions')(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compilar com metricas apropriadas
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=[
                'binary_accuracy',
                tf.keras.metrics.AUC(name='auc', multi_label=True),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        self.logger.info(f"Modelo criado com {self.model.count_params():,} parametros")
        
    def calculate_class_weights(self):
        """Calcula pesos para classes desbalanceadas"""
        weights = {}
        for i, pathology in enumerate(self.pathologies):
            pos_count = self.train_df[pathology].sum()
            neg_count = len(self.train_df) - pos_count
            
            if pos_count > 0:
                weight_pos = (1 / pos_count) * (len(self.train_df) / 2.0)
                weight_neg = (1 / neg_count) * (len(self.train_df) / 2.0)
                weights[i] = {0: weight_neg, 1: weight_pos}
            else:
                weights[i] = {0: 1.0, 1: 1.0}
                
        return weights
        
    def train(self, epochs=30):
        """Treina o modelo com estrategias avancadas"""
        self.logger.info("\nIniciando treinamento...")
        
        # Callbacks
        callbacks = [
            # Early stopping baseado em AUC
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=5,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduzir learning rate
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                mode='max',
                verbose=1
            ),
            
            # Salvar melhor modelo
            tf.keras.callbacks.ModelCheckpoint(
                'best_nih_model.h5',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # TensorBoard
            tf.keras.callbacks.TensorBoard(
                log_dir=f'logs/tensorboard_{self.start_time.strftime("%Y%m%d_%H%M%S")}',
                histogram_freq=1
            )
        ]
        
        # Treinar
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        self.save_training_plots(history)
        return history
        
    def save_training_plots(self, history):
        """Salva graficos de treinamento"""
        plt.figure(figsize=(15, 5))
        
        # AUC
        plt.subplot(1, 3, 1)
        plt.plot(history.history['auc'], label='Train AUC')
        plt.plot(history.history['val_auc'], label='Val AUC')
        plt.title('Model AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        
        # Loss
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Precision/Recall
        plt.subplot(1, 3, 3)
        plt.plot(history.history['precision'], label='Train Precision')
        plt.plot(history.history['recall'], label='Train Recall')
        plt.plot(history.history['val_precision'], label='Val Precision')
        plt.plot(history.history['val_recall'], label='Val Recall')
        plt.title('Precision and Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300)
        plt.close()
        
    def evaluate_model(self):
        """Avalia modelo no conjunto de teste"""
        self.logger.info("\nAvaliando modelo no conjunto de teste...")
        
        # Criar gerador de teste
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = self.create_test_generator(self.test_df, test_datagen)
        
        # Avaliar
        results = self.model.evaluate(
            test_generator,
            steps=len(self.test_df) // self.batch_size
        )
        
        # Metricas por patologia
        self.logger.info("\nMetricas por patologia:")
        y_true, y_pred = self.get_predictions(test_generator)
        
        for i, pathology in enumerate(self.pathologies):
            if y_true[:, i].sum() > 0:  # Apenas se houver casos
                auc = tf.keras.metrics.AUC()
                auc.update_state(y_true[:, i], y_pred[:, i])
                
                self.logger.info(f"{pathology}: AUC = {auc.result().numpy():.3f}")
                
    def run(self):
        """Executa pipeline completo"""
        try:
            # Carregar dados
            if not self.load_metadata():
                return
                
            # Criar splits
            self.create_data_splits()
            
            # Criar geradores
            self.create_generators()
            
            # Criar modelo
            self.create_model()
            
            # Treinar
            self.train(epochs=20)
            
            # Avaliar
            self.evaluate_model()
            
            # Relatorio final
            self.logger.info("\nTreinamento concluido com sucesso!")
            self.logger.info(f"Modelo salvo em: best_nih_model.h5")
            self.logger.info(f"Tempo total: {datetime.now() - self.start_time}")
            
        except Exception as e:
            self.logger.error(f"Erro: {str(e)}", exc_info=True)
        finally:
            # Liberar memoria
            tf.keras.backend.clear_session()
            gc.collect()

# Executar
if __name__ == "__main__":
    print("Sistema Otimizado de Treinamento NIH Chest X-ray")
    print("="*50)
    
    trainer = NIHXrayTrainer(
        data_path='D:/NIH_CHEST_XRAY/',
        batch_size=16  # Reduzido para economizar RAM
    )
    trainer.run()