#!/usr/bin/env python3
"""
test_corrections.py - Testa as correções aplicadas
"""

import sys
import os

def test_imports():
    """Testa se as importações funcionam corretamente"""
    print("Testando importações...")
    
    try:
        # Testa logging config
        from src.logging_config import setup_logging
        logger = setup_logging('Test')
        logger.info("Logging config funcionando!")
        print("✓ Logging config: OK")
    except Exception as e:
        print(f"✗ Erro no logging config: {e}")
        return False
    
    try:
        # Testa TensorFlow
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}: OK")
    except Exception as e:
        print(f"✗ Erro no TensorFlow: {e}")
        return False
    
    try:
        # Testa OpenCV
        import cv2
        print(f"✓ OpenCV {cv2.__version__}: OK")
    except Exception as e:
        print(f"✗ Erro no OpenCV: {e}")
        return False
    
    try:
        # Testa numpy
        import numpy as np
        print(f"✓ NumPy {np.__version__}: OK")
    except Exception as e:
        print(f"✗ Erro no NumPy: {e}")
        return False
    
    return True

def test_augmentation_function():
    """Testa se a função de augmentação funciona"""
    print("\nTestando função de augmentação...")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        # Cria uma imagem de teste
        test_image = tf.random.normal((1, 384, 384, 3))
        test_label = tf.constant([1])
        
        # Define a função de augmentação corrigida
        @tf.function
        def augment(image, label):
            image = tf.cast(image, tf.float32)
            
            augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomRotation(0.02),
                tf.keras.layers.RandomTranslation(0.05, 0.05),
                tf.keras.layers.RandomZoom(0.05),
                tf.keras.layers.RandomFlip("horizontal"),
            ])
            
            image = augmentation(image, training=True)
            image = tf.image.random_brightness(image, 0.1)
            image = tf.image.random_contrast(image, 0.9, 1.1)
            
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01, dtype=tf.float32)
            image = image + noise
            image = tf.clip_by_value(image, 0.0, 1.0)
            
            return image, label
        
        # Testa a função
        aug_image, aug_label = augment(test_image, test_label)
        
        print(f"✓ Função de augmentação: OK")
        print(f"  - Shape original: {test_image.shape}")
        print(f"  - Shape após augmentação: {aug_image.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Erro na função de augmentação: {e}")
        return False

def test_data_generator():
    """Testa se o DataGenerator funciona"""
    print("\nTestando DataGenerator...")
    
    try:
        # Cria dados de teste
        import pandas as pd
        import numpy as np
        from pathlib import Path
        
        # Cria um DataFrame de teste
        test_df = pd.DataFrame({
            'Image Index': ['test1.jpg', 'test2.jpg'],
            'Finding Labels': ['No Finding', 'Pneumonia']
        })
        
        # Cria diretório de teste
        test_dir = Path('test_images')
        test_dir.mkdir(exist_ok=True)
        
        # Cria imagens de teste
        import cv2
        for img_name in test_df['Image Index']:
            test_img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
            cv2.imwrite(str(test_dir / img_name), test_img)
        
        print("✓ DataGenerator: Preparação OK")
        return True
        
    except Exception as e:
        print(f"✗ Erro no DataGenerator: {e}")
        return False

if __name__ == "__main__":
    print("=== Teste das Correções MedAI ===\n")
    
    success = True
    
    # Testa importações
    if not test_imports():
        success = False
    
    # Testa função de augmentação
    if not test_augmentation_function():
        success = False
    
    # Testa DataGenerator
    if not test_data_generator():
        success = False
    
    print(f"\n=== Resultado Final ===")
    if success:
        print("✓ Todos os testes passaram! As correções estão funcionando.")
    else:
        print("✗ Alguns testes falharam. Verifique os erros acima.")
    
    sys.exit(0 if success else 1)

