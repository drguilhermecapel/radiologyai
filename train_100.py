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
from sklearn.metrics import roc_auc_score, f1_score
import json
import warnings
warnings.filterwarnings('ignore')

# Fix para encoding no Windows
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

class NIHXrayTrainerFixed:
    def __init__(self, data_path='D:/NIH_CHEST_XRAY/', batch_size=16):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.image_size = (224, 224)  # Reduzido para economizar memória
        self.num_classes = 14
        self.start_time = datetime.now()
        
        self.pathologies = [
            'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
            'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
            'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
        ]
        
        self.setup_logging()
        self.configure_tensorflow()
        
    def setup_logging(self):
        """Configura logging com encoding correto"""
        os.makedirs('logs', exist_ok=True)
        log_file = f'logs/nih_training_{self.start_time.strftime("%Y%m%d_%H%M%S")}.log'
        
        # Handler com encoding UTF-8
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        console_handler = logging.StreamHandler(sys.stdout)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[file_handler, console_handler]
        )
        
        self.logger = logging.getLogger('NIH-Fixed')
        self.logger.info("Sistema NIH Chest X-ray - Versao Corrigida")
        
    def configure_tensorflow(self):
        """Configura TensorFlow"""
        # Desabilitar mixed precision por enquanto para evitar problemas
        # policy = tf.keras.mixed_precision.Policy('float32')
        # tf.keras.mixed_precision.set_global_policy(policy)
        
        # GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self.logger.info(f"GPU encontrada: {gpus[0].name}")
        else:
            self.logger.info("Usando CPU")
            
    def find_image_locations(self):
        """Mapeia a localização real de cada imagem"""
        self.logger.info("Mapeando localizacao das imagens...")
        
        # Criar dicionário de localização das imagens
        self.image_locations = {}
        
        # Procurar em todos os subdiretórios images_*
        image_dirs = list(self.data_path.glob('images_*'))
        
        if not image_dirs:
            # Tentar diretório único 'images'
            single_dir = self.data_path / 'images'
            if single_dir.exists():
                image_dirs = [single_dir]
            else:
                self.logger.error("Nenhum diretorio de imagens encontrado!")
                return False
                
        total_images = 0
        for img_dir in image_dirs:
            # Procurar em subdirs também
            for subdir in img_dir.rglob('*'):
                if subdir.is_dir():
                    png_files = list(subdir.glob('*.png'))
                    for png_file in png_files:
                        self.image_locations[png_file.name] = str(png_file)
                        total_images += 1
                        
        self.logger.info(f"Total de imagens encontradas: {total_images}")
        self.logger.info(f"Diretorios de imagens: {[str(d) for d in image_dirs]}")
        
        return total_images > 0
        
    def load_and_prepare_data(self):
        """Carrega dados e mapeia caminhos corretos"""
        csv_path = self.data_path / 'Data_Entry_2017_v2020.csv'
        
        if not csv_path.exists():
            # Tentar nome alternativo
            csv_path = self.data_path / 'Data_Entry_2017.csv'
            
        self.logger.info(f"Carregando metadados de {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # Mapear localizações das imagens
        if not self.find_image_locations():
            return False
            
        # Verificar quais imagens do CSV existem fisicamente
        self.logger.info("Verificando disponibilidade das imagens...")
        
        # Adicionar caminho completo para cada imagem
        valid_images = []
        missing_count = 0
        
        for idx, row in self.df.iterrows():
            img_name = row['Image Index']
            if img_name in self.image_locations:
                valid_images.append(idx)
            else:
                missing_count += 1
                
        self.logger.info(f"Imagens validas: {len(valid_images)}")
        self.logger.info(f"Imagens faltando: {missing_count}")
        
        # Filtrar apenas imagens válidas
        self.df = self.df.loc[valid_images].reset_index(drop=True)
        
        # Adicionar coluna com caminho completo
        self.df['full_path'] = self.df['Image Index'].apply(
            lambda x: self.image_locations.get(x, '')
        )
        
        # Processar labels
        for pathology in self.pathologies:
            self.df[pathology] = self.df['Finding Labels'].apply(
                lambda x: 1 if pathology in x else 0
            )
            
        # Limitar dataset para teste
        max_samples = min(10000, len(self.df))  # Usar menos amostras inicialmente
        self.logger.info(f"Usando {max_samples} imagens para teste rapido")
        self.df = self.df.sample(n=max_samples, random_state=42)
        
        # Estatísticas
        self.logger.info("\nDistribuicao de patologias:")
        for pathology in self.pathologies:
            count = self.df[pathology].sum()
            percentage = (count / len(self.df)) * 100
            self.logger.info(f"{pathology}: {count} ({percentage:.1f}%)")
            
        return True
        
    def preprocess_image(self, image_path, label):
        """Preprocessa imagem com tratamento de erros"""
        try:
            # Converter string para tensor se necessário
            if isinstance(image_path, bytes):
                image_path = image_path.decode('utf-8')
            elif tf.is_tensor(image_path):
                image_path = image_path.numpy().decode('utf-8')
                
            # Ler arquivo
            image = tf.io.read_file(image_path)
            
            # Decodificar PNG
            image = tf.image.decode_png(image, channels=1)
            
            # Converter para float32
            image = tf.cast(image, tf.float32)
            
            # Resize
            image = tf.image.resize(image, self.image_size)
            
            # Normalizar para [0, 1]
            image = image / 255.0
            
            # Aplicar CLAHE-like enhancement
            mean = tf.reduce_mean(image)
            std = tf.math.reduce_std(image)
            image = (image - mean) / (std + 1e-7)
            image = tf.clip_by_value(image, -2, 2)
            image = (image + 2) / 4
            
            # Converter para 3 canais
            image = tf.repeat(image, 3, axis=-1)
            
            return image, label
            
        except Exception as e:
            # Em caso de erro, retornar imagem preta
            self.logger.warning(f"Erro ao processar {image_path}: {str(e)}")
            blank_image = tf.zeros((*self.image_size, 3))
            return blank_image, label
            
    def create_simple_model(self):
        """Cria modelo mais simples para evitar problemas de memória"""
        inputs = tf.keras.Input(shape=(*self.image_size, 3))
        
        # Base model menor - MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Congelar base
        base_model.trainable = False
        
        # Construir modelo
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        return model
        
    def create_data_generator_safe(self, df, is_training=True):
        """Cria gerador de dados com tratamento de erros"""
        # Usar caminhos completos
        image_paths = df['full_path'].values
        labels = df[self.pathologies].values.astype(np.float32)
        
        def generator():
            indices = np.arange(len(image_paths))
            if is_training:
                np.random.shuffle(indices)
                
            for idx in indices:
                try:
                    # Carregar e preprocessar imagem
                    img_path = image_paths[idx]
                    label = labels[idx]
                    
                    # Verificar se arquivo existe
                    if os.path.exists(img_path):
                        image, label = self.preprocess_image(img_path, label)
                        yield image, label
                    else:
                        # Pular imagem faltante
                        continue
                        
                except Exception as e:
                    self.logger.warning(f"Erro no gerador: {str(e)}")
                    continue
                    
        # Criar dataset a partir do gerador
        output_signature = (
            tf.TensorSpec(shape=(*self.image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(self.num_classes,), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        if is_training:
            dataset = dataset.repeat()
            
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
        
    def train_simple_model(self, epochs=10):
        """Treina modelo simples com configurações seguras"""
        self.logger.info("\nIniciando treinamento simplificado...")
        
        # Criar splits
        train_df = self.df.sample(frac=0.8, random_state=42)
        val_df = self.df[~self.df.index.isin(train_df.index)]
        
        self.logger.info(f"Treino: {len(train_df)} imagens")
        self.logger.info(f"Validacao: {len(val_df)} imagens")
        
        # Criar geradores seguros
        train_data = self.create_data_generator_safe(train_df, is_training=True)
        val_data = self.create_data_generator_safe(val_df, is_training=False)
        
        # Criar modelo
        model = self.create_simple_model()
        
        # Compilar
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                'binary_accuracy',
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        # Callbacks básicos
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'best_model_simple.h5',
                monitor='val_auc',
                save_best_only=True,
                mode='max'
            ),
            
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=3,
                mode='max',
                restore_best_weights=True
            ),
            
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Calcular steps
        steps_per_epoch = len(train_df) // self.batch_size
        validation_steps = len(val_df) // self.batch_size
        
        # Treinar
        try:
            history = model.fit(
                train_data,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_data,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
            
            return model, history
            
        except Exception as e:
            self.logger.error(f"Erro durante treinamento: {str(e)}")
            return None, None
            
    def evaluate_model(self, model, test_df):
        """Avalia modelo com métricas básicas"""
        self.logger.info("\nAvaliando modelo...")
        
        # Criar gerador de teste
        test_data = self.create_data_generator_safe(test_df, is_training=False)
        test_labels = test_df[self.pathologies].values
        
        # Fazer predições
        predictions = []
        steps = len(test_df) // self.batch_size
        
        for i, (images, _) in enumerate(test_data.take(steps)):
            preds = model.predict(images, verbose=0)
            predictions.append(preds)
            
        predictions = np.concatenate(predictions)
        
        # Calcular AUC por patologia
        results = {}
        for i, pathology in enumerate(self.pathologies):
            if test_labels[:len(predictions), i].sum() > 0:
                auc = roc_auc_score(
                    test_labels[:len(predictions), i], 
                    predictions[:, i]
                )
                results[pathology] = auc
                self.logger.info(f"{pathology}: AUC = {auc:.4f}")
                
        mean_auc = np.mean(list(results.values()))
        self.logger.info(f"\nAUC Medio: {mean_auc:.4f}")
        
        return results, mean_auc
        
    def run(self):
        """Pipeline simplificado e seguro"""
        try:
            # Carregar dados
            if not self.load_and_prepare_data():
                self.logger.error("Falha ao carregar dados")
                return
                
            # Treinar modelo simples
            model, history = self.train_simple_model(epochs=5)  # Poucas épocas para teste
            
            if model is None:
                self.logger.error("Falha no treinamento")
                return
                
            # Avaliar
            test_df = self.df.sample(frac=0.1, random_state=42)
            results, mean_auc = self.evaluate_model(model, test_df)
            
            # Salvar resultados
            output = {
                'timestamp': datetime.now().isoformat(),
                'mean_auc': float(mean_auc),
                'pathology_results': {k: float(v) for k, v in results.items()},
                'training_time': str(datetime.now() - self.start_time)
            }
            
            with open('training_results_simple.json', 'w') as f:
                json.dump(output, f, indent=2)
                
            self.logger.info("\n" + "="*50)
            self.logger.info("Treinamento Concluido!")
            self.logger.info(f"AUC Medio: {mean_auc:.4f}")
            self.logger.info("="*50)
            
        except Exception as e:
            self.logger.error(f"Erro fatal: {str(e)}", exc_info=True)
        finally:
            gc.collect()
            tf.keras.backend.clear_session()

# Executar
if __name__ == "__main__":
    print("="*60)
    print("NIH CHEST X-RAY - VERSAO CORRIGIDA")
    print("="*60)
    print("Correcoes implementadas:")
    print("1. Mapeamento correto dos diretorios de imagens")
    print("2. Tratamento de erros de encoding")
    print("3. Modelo simplificado para evitar problemas de memoria")
    print("4. Geradores de dados com tratamento de erros")
    print("5. Pipeline mais robusto")
    print("="*60)
    
    trainer = NIHXrayTrainerFixed(
        data_path='D:/NIH_CHEST_XRAY/',
        batch_size=16
    )
    
    trainer.run()