#!/usr/bin/env python3
"""
Create synthetic test images for validating AI fixes
"""

import numpy as np
from PIL import Image
import os

def create_normal_chest_image():
    """Create a synthetic normal chest X-ray image"""
    img = np.zeros((512, 512), dtype=np.uint8)
    
    img[100:400, 50:250] = 40   # Left lung
    img[100:400, 270:470] = 40  # Right lung
    
    for i in range(8):
        y = 120 + i * 30
        img[y:y+3, 50:470] = 120
    
    img[200:350, 200:320] = 80
    
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    return img

def create_pleural_effusion_image():
    """Create a synthetic chest X-ray with pleural effusion"""
    img = create_normal_chest_image()
    
    img[350:450, 50:250] = 180   # Left pleural effusion
    
    for x in range(50, 250):
        curve_height = int(20 * np.sin((x - 50) * np.pi / 200))
        img[350 - curve_height:350, x] = 160
    
    return img

def main():
    """Create test images for validation"""
    print("Creating synthetic test images...")
    
    test_dir = "/home/ubuntu/repos/radiologyai/test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    normal_img = create_normal_chest_image()
    normal_pil = Image.fromarray(normal_img, mode='L')
    normal_path = os.path.join(test_dir, "normal_chest.png")
    normal_pil.save(normal_path)
    print(f"✅ Created normal chest image: {normal_path}")
    
    effusion_img = create_pleural_effusion_image()
    effusion_pil = Image.fromarray(effusion_img, mode='L')
    effusion_path = os.path.join(test_dir, "pleural_effusion_chest.png")
    effusion_pil.save(effusion_path)
    print(f"✅ Created pleural effusion image: {effusion_path}")
    
    print("Test images created successfully!")
    return normal_path, effusion_path

if __name__ == "__main__":
    main()
