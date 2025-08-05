#!/usr/bin/env python3
"""
Askeri Hava Aracı Tespiti Modeli Eğitimi

Bu script, YOLO formatındaki askeri hava aracı verisetini kullanarak
bir nesne tespit modeli eğitir.

Veriset yapısı:
- 43 farklı askeri hava aracı sınıfı
- Eğitim: ~9,431 görüntü
- Doğrulama: ~2,359 görüntü
- YOLO formatında etiketler (class x_center y_center width height)
"""

import os
import torch
import yaml
from pathlib import Path
import argparse
import logging
from datetime import datetime
import shutil

# YOLOv8 için ultralytics import
try:
    from ultralytics import YOLO
    print("[+] Ultralytics YOLOv8 kutuphanesi bulundu")
except ImportError:
    print("[X] Ultralytics YOLOv8 kutuphanesi bulunamadi!")
    print("Kurulum icin: pip install ultralytics")
    exit(1)

def setup_logging():
    """Logging yapılandırması"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', 
                              encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_dataset_structure(data_path):
    """Veriset yapısını doğrula"""
    logger = logging.getLogger(__name__)
    
    # Gerekli dizinleri kontrol et
    required_dirs = [
        'images/aircraft_train',
        'images/aircraft_val', 
        'labels/aircraft_train',
        'labels/aircraft_val'
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join(data_path, dir_path)
        if not os.path.exists(full_path):
            logger.error(f"[X] Gerekli dizin bulunamadi: {full_path}")
            return False
        logger.info(f"[+] Dizin bulundu: {dir_path}")
    
    # Dosya sayılarını kontrol et
    train_images = len(os.listdir(os.path.join(data_path, 'images/aircraft_train')))
    val_images = len(os.listdir(os.path.join(data_path, 'images/aircraft_val')))
    train_labels = len(os.listdir(os.path.join(data_path, 'labels/aircraft_train')))
    val_labels = len(os.listdir(os.path.join(data_path, 'labels/aircraft_val')))
    
    logger.info(f"[#] Veriset istatistikleri:")
    logger.info(f"   - Egitim goruntuleri: {train_images}")
    logger.info(f"   - Egitim etiketleri: {train_labels}")
    logger.info(f"   - Dogrulama goruntuleri: {val_images}")
    logger.info(f"   - Dogrulama etiketleri: {val_labels}")
    
    return True

def create_yolo_config(data_path, output_path='aircraft_config.yaml'):
    """YOLO eğitimi için yapılandırma dosyası oluştur"""
    logger = logging.getLogger(__name__)
    
    # Sınıf isimlerini aircraft_names.yaml'dan oku
    names_file = os.path.join(data_path, 'aircraft_names.yaml')
    with open(names_file, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # YOLO yapılandırması
    yolo_config = {
        'path': os.path.abspath(data_path),
        'train': 'images/aircraft_train',
        'val': 'images/aircraft_val',
        'test': '',  # Test seti yok
        'names': config_data['names']
    }
    
    # Yapılandırmayı kaydet
    config_path = os.path.join(data_path, output_path)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(yolo_config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"[+] YOLO yapilandirma dosyasi olusturuldu: {config_path}")
    logger.info(f"[#] Toplam sinif sayisi: {len(config_data['names'])}")
    
    return config_path

def train_model(config_path, model_size='yolov8n', epochs=100, imgsz=640, batch_size=16):
    """YOLO modelini eğit"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"[#] Model egitimi baslatilyior...")
    logger.info(f"   - Model boyutu: {model_size}")
    logger.info(f"   - Epoch sayisi: {epochs}")
    logger.info(f"   - Goruntu boyutu: {imgsz}")
    logger.info(f"   - Batch boyutu: {batch_size}")
    
    # Model yükle
    model = YOLO(f'{model_size}.pt')
    
    # GPU kontrol
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"[PC] Kullanilan cihaz: {device}")
    
    # Eğitim parametreleri
    train_args = {
        'data': config_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': device,
        'project': 'military_aircraft_detection',
        'name': f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'save': True,
        'save_period': 10,  
        'cache': True,  
        'patience': 50,  
        'plots': True, 
        'verbose': True
    }
    
    try:
        # Egitimi baslat
        results = model.train(**train_args)
        logger.info("[+] Model egitimi tamamlandi!")
        return results
        
    except Exception as e:
        logger.error(f"[X] Egitim hatasi: {str(e)}")
        return None

def evaluate_model(model_path, config_path):
    """Eğitilen modeli değerlendir"""
    logger = logging.getLogger(__name__)
    
    logger.info("[#] Model degerlendirmesi baslatilyior...")
    
    try:
        # En iyi modeli yukle
        model = YOLO(model_path)
        
        # Dogrulama setinde degerlendir
        results = model.val(data=config_path)
        
        logger.info("[+] Model degerlendirmesi tamamlandi!")
        logger.info(f"[#] mAP50: {results.box.map50:.4f}")
        logger.info(f"[#] mAP50-95: {results.box.map:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"[X] Degerlendirme hatasi: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Askeri Hava Aracı Tespiti Modeli Eğitimi')
    parser.add_argument('--data', type=str, default='.',
                       help='Veriset dizini yolu (varsayılan: mevcut dizin)')
    parser.add_argument('--model', type=str, default='yolov8n',
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                       help='YOLOv8 model boyutu (varsayılan: yolov8n)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Eğitim epoch sayısı (varsayılan: 100)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Giriş görüntü boyutu (varsayılan: 640)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch boyutu (varsayılan: 16)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Sadece veriset doğrulaması yap')
    
    args = parser.parse_args()
    
    # Logging ayarla
    logger = setup_logging()
    logger.info("[#] Askeri Hava Araci Tespiti Modeli Egitimi Baslatilyior")
    
    # Veriset yolunu ayarla
    data_path = os.path.abspath(args.data)
    logger.info(f"[F] Veriset dizini: {data_path}")
    
    # Veriset yapısını doğrula
    if not validate_dataset_structure(data_path):
        logger.error("[X] Veriset dogrulamasi basarisiz!")
        return
    
    if args.validate_only:
        logger.info("[+] Veriset dogrulamasi tamamlandi. Sadece dogrulama modunda calisildi.")
        return
    
    # YOLO yapılandırma dosyası oluştur
    config_path = create_yolo_config(data_path)
    
    # Modeli eğit
    results = train_model(
        config_path=config_path,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch
    )
    
    if results:
        # En iyi model yolunu bul
        project_dir = os.path.join('military_aircraft_detection')
        if os.path.exists(project_dir):
            # En son eğitim klasörünü bul
            train_dirs = [d for d in os.listdir(project_dir) if d.startswith('train_')]
            if train_dirs:
                latest_train = sorted(train_dirs)[-1]
                best_model_path = os.path.join(project_dir, latest_train, 'weights', 'best.pt')
                
                if os.path.exists(best_model_path):
                    logger.info(f"[+] En iyi model: {best_model_path}")
                    
                    # Modeli değerlendir
                    evaluate_model(best_model_path, config_path)
                    
                    # Model dosyalarını kopyala
                    shutil.copy2(best_model_path, os.path.join(data_path, 'military_aircraft_best.pt'))
                    logger.info(f"[+] Model kaydedildi: military_aircraft_best.pt")
    
    logger.info("[+] Egitim sureci tamamlandi!")

if __name__ == '__main__':
    main()
