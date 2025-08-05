#!/usr/bin/env python3
"""
Askeri Hava Aracı Tespiti - Test ve Inference

Bu script, eğitilmiş modeli kullanarak yeni görüntüler üzerinde
askeri hava aracı tespiti yapar.
"""

import os
import cv2
import torch
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

try:
    from ultralytics import YOLO
    print("✓ Ultralytics YOLOv8 kütüphanesi bulundu")
except ImportError:
    print("❌ Ultralytics YOLOv8 kütüphanesi bulunamadı!")
    print("Kurulum için: pip install ultralytics")
    exit(1)

# Askeri hava aracı sınıf isimleri
AIRCRAFT_CLASSES = {
    0: 'A10', 1: 'A400M', 2: 'AG600', 3: 'AV8B', 4: 'B1', 5: 'B2', 6: 'B52',
    7: 'Be200', 8: 'C130', 9: 'C17', 10: 'C2', 11: 'C5', 12: 'E2', 13: 'E7',
    14: 'EF2000', 15: 'F117', 16: 'F14', 17: 'F15', 18: 'F16', 19: 'F18',
    20: 'F22', 21: 'F35', 22: 'F4', 23: 'J20', 24: 'JAS39', 25: 'MQ9',
    26: 'Mig31', 27: 'Mirage2000', 28: 'P3', 29: 'RQ4', 30: 'Rafale',
    31: 'SR71', 32: 'Su34', 33: 'Su57', 34: 'Tornado', 35: 'Tu160',
    36: 'Tu95', 37: 'U2', 38: 'US2', 39: 'V22', 40: 'Vulcan', 41: 'XB70', 42: 'YF23'
}

def setup_logging():
    """Logging yapılandırması"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def load_model(model_path):
    """Eğitilmiş modeli yükle"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model dosyası bulunamadı: {model_path}")
        return None
    
    try:
        model = YOLO(model_path)
        logger.info(f"✅ Model yüklendi: {model_path}")
        return model
    except Exception as e:
        logger.error(f"❌ Model yükleme hatası: {str(e)}")
        return None

def predict_image(model, image_path, confidence_threshold=0.5):
    """Tek bir görüntü üzerinde tahmin yap"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(image_path):
        logger.error(f"❌ Görüntü dosyası bulunamadı: {image_path}")
        return None
    
    try:
        # Tahmin yap
        results = model(image_path, conf=confidence_threshold)
        logger.info(f"✅ Tahmin tamamlandı: {image_path}")
        return results[0]  # İlk (ve tek) sonuç
    except Exception as e:
        logger.error(f"❌ Tahmin hatası: {str(e)}")
        return None

def visualize_predictions(image_path, results, output_path=None, show_labels=True):
    """Tahmin sonuçlarını görselleştir"""
    logger = logging.getLogger(__name__)
    
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Matplotlib ile görselleştir
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    # Tespit edilen nesneleri çiz
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        
        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Rastgele renk seç
            color = plt.cm.Set3(cls / len(AIRCRAFT_CLASSES))
            
            # Bounding box çiz
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Etiket ekle
            if show_labels:
                aircraft_name = AIRCRAFT_CLASSES.get(cls, f'Class_{cls}')
                label = f'{aircraft_name}: {conf:.2f}'
                ax.text(x1, y1-10, label, color=color, fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        logger.info(f"📊 Tespit edilen nesne sayısı: {len(boxes)}")
    else:
        logger.info("❌ Hiç nesne tespit edilmedi")
    
    ax.axis('off')
    ax.set_title(f'Askeri Hava Aracı Tespiti - {os.path.basename(image_path)}', fontsize=14)
    
    # Kaydet veya göster
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        logger.info(f"✅ Sonuç kaydedildi: {output_path}")
    else:
        plt.show()
    
    plt.close()

def batch_predict(model, input_dir, output_dir, confidence_threshold=0.5):
    """Bir dizindeki tüm görüntüler üzerinde tahmin yap"""
    logger = logging.getLogger(__name__)
    
    # Desteklenen görüntü formatları
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    # Giriş dizinindeki görüntüleri bul
    image_files = []
    for ext in supported_formats:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        logger.error(f"❌ {input_dir} dizininde desteklenen görüntü bulunamadı")
        return
    
    logger.info(f"📁 {len(image_files)} görüntü bulundu")
    
    # Çıkış dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Her görüntü için tahmin yap
    for i, image_file in enumerate(image_files):
        logger.info(f"🔍 İşleniyor ({i+1}/{len(image_files)}): {image_file.name}")
        
        # Tahmin yap
        results = predict_image(model, str(image_file), confidence_threshold)
        
        if results is not None:
            # Çıkış dosya yolu
            output_file = os.path.join(output_dir, f'predicted_{image_file.name}')
            
            # Görselleştir ve kaydet
            visualize_predictions(str(image_file), results, output_file)
    
    logger.info(f"✅ Toplu tahmin tamamlandı. Sonuçlar: {output_dir}")

def get_model_info(model):
    """Model hakkında bilgi ver"""
    logger = logging.getLogger(__name__)
    
    logger.info("📋 Model Bilgileri:")
    logger.info(f"   - Model türü: {type(model).__name__}")
    logger.info(f"   - Sınıf sayısı: {len(model.names)}")
    logger.info(f"   - Sınıf isimleri: {list(model.names.values())}")
    
    # GPU kullanımı
    device = next(model.model.parameters()).device
    logger.info(f"   - Cihaz: {device}")

def main():
    parser = argparse.ArgumentParser(description='Askeri Hava Aracı Tespiti - Test ve Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Eğitilmiş model dosyası yolu (.pt)')
    parser.add_argument('--input', type=str, required=True,
                       help='Giriş görüntü dosyası veya dizini')
    parser.add_argument('--output', type=str, 
                       help='Çıkış dizini (belirtilmezse görüntü gösterilir)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Güven eşiği (varsayılan: 0.5)')
    parser.add_argument('--batch', action='store_true',
                       help='Dizindeki tüm görüntüleri işle')
    
    args = parser.parse_args()
    
    # Logging ayarla
    logger = setup_logging()
    logger.info("🎯 Askeri Hava Aracı Tespiti - Test Modu")
    
    # Modeli yükle
    model = load_model(args.model)
    if model is None:
        return
    
    # Model bilgilerini göster
    get_model_info(model)
    
    # Giriş kontrolü
    if not os.path.exists(args.input):
        logger.error(f"❌ Giriş bulunamadı: {args.input}")
        return
    
    # Toplu işlem mi tek dosya mı?
    if args.batch or os.path.isdir(args.input):
        if not args.output:
            logger.error("❌ Toplu işlem için çıkış dizini gerekli (--output)")
            return
        
        batch_predict(model, args.input, args.output, args.confidence)
    else:
        # Tek görüntü işle
        results = predict_image(model, args.input, args.confidence)
        
        if results is not None:
            output_path = None
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                filename = os.path.basename(args.input)
                output_path = os.path.join(args.output, f'predicted_{filename}')
            
            visualize_predictions(args.input, results, output_path)
    
    logger.info("🎉 İşlem tamamlandı!")

if __name__ == '__main__':
    main()
