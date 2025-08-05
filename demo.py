#!/usr/bin/env python3
"""
Askeri Hava Aracı Tespiti - Demo Script

Bu script, eğitilmiş modeli kullanarak hızlı bir demo gösterir.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

def check_dependencies():
    """Gerekli kütüphaneleri kontrol et"""
    required_packages = {
        'torch': 'torch>=2.0.0',
        'ultralytics': 'ultralytics>=8.0.0',
        'cv2': 'opencv-python>=4.7.0',
        'matplotlib': 'matplotlib>=3.5.0',
        'yaml': 'PyYAML>=6.0'
    }
    
    missing_packages = []
    
    for package, version_info in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} bulundu")
        except ImportError:
            missing_packages.append(version_info)
            print(f"❌ {package} bulunamadı")
    
    if missing_packages:
        print("\n🔧 Eksik kütüphaneleri yüklemek için:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_demo_environment():
    """Demo ortamını hazırla"""
    print("🎬 Demo ortamı hazırlanıyor...")
    
    # Çıkış dizinini oluştur
    demo_output = "demo_results"
    os.makedirs(demo_output, exist_ok=True)
    
    return demo_output

def run_training_demo(quick_mode=True):
    """Eğitim demo'sunu çalıştır"""
    print("\n🚀 Eğitim Demo'su başlatılıyor...")
    
    if quick_mode:
        epochs = 10
        batch_size = 8
        model_size = 'yolov8n'
        print(f"⚡ Hızlı mod: {epochs} epoch, batch={batch_size}, model={model_size}")
    else:
        epochs = 50
        batch_size = 16
        model_size = 'yolov8s'
        print(f"🎯 Standart mod: {epochs} epoch, batch={batch_size}, model={model_size}")
    
    # Eğitim komutu
    train_cmd = [
        sys.executable, "train_military_aircraft_detection.py",
        "--model", model_size,
        "--epochs", str(epochs),
        "--batch", str(batch_size),
        "--imgsz", "640"
    ]
    
    print(f"🔨 Komut: {' '.join(train_cmd)}")
    
    try:
        import subprocess
        result = subprocess.run(train_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Eğitim demo'su başarıyla tamamlandı!")
            return True
        else:
            print(f"❌ Eğitim hatası: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Demo çalıştırma hatası: {str(e)}")
        return False

def run_inference_demo():
    """Inference demo'sunu çalıştır"""
    print("\n🔍 Inference Demo'su başlatılıyor...")
    
    # En iyi modeli bul
    model_path = None
    possible_models = [
        "military_aircraft_best.pt",
        "military_aircraft_detection/train_*/weights/best.pt",
        "runs/detect/train/weights/best.pt"
    ]
    
    for pattern in possible_models:
        if '*' in pattern:
            # Glob pattern
            from pathlib import Path
            matches = list(Path('.').glob(pattern))
            if matches:
                model_path = str(matches[-1])  # En son eğitilen
                break
        else:
            if os.path.exists(pattern):
                model_path = pattern
                break
    
    if not model_path:
        print("❌ Eğitilmiş model bulunamadı. Önce eğitim yapın.")
        return False
    
    print(f"📦 Model bulundu: {model_path}")
    
    # Test görüntülerini bul
    test_images = []
    image_dirs = ["images/aircraft_val", "images/aircraft_train"]
    
    for img_dir in image_dirs:
        if os.path.exists(img_dir):
            images = list(Path(img_dir).glob("*.jpg"))[:5]  # İlk 5 görüntü
            test_images.extend(images)
            break
    
    if not test_images:
        print("❌ Test görüntüleri bulunamadı")
        return False
    
    print(f"🖼️ {len(test_images)} test görüntüsü bulundu")
    
    # Demo çıkış dizini
    demo_output = setup_demo_environment()
    
    # Inference komutları
    for i, img_path in enumerate(test_images):
        print(f"\n🎯 Test {i+1}/{len(test_images)}: {img_path.name}")
        
        test_cmd = [
            sys.executable, "test_military_aircraft_detection.py",
            "--model", model_path,
            "--input", str(img_path),
            "--output", demo_output,
            "--confidence", "0.3"
        ]
        
        try:
            import subprocess
            result = subprocess.run(test_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Test {i+1} tamamlandı")
            else:
                print(f"❌ Test {i+1} hatası: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Test çalıştırma hatası: {str(e)}")
    
    print(f"\n🎉 Inference demo'su tamamlandı! Sonuçlar: {demo_output}/")
    return True

def run_validation_demo():
    """Veriset doğrulama demo'sunu çalıştır"""
    print("\n📋 Veriset Doğrulama Demo'su...")
    
    validation_cmd = [
        sys.executable, "train_military_aircraft_detection.py",
        "--validate-only"
    ]
    
    try:
        import subprocess
        result = subprocess.run(validation_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Veriset doğrulaması başarılı!")
            print(result.stdout)
            return True
        else:
            print(f"❌ Doğrulama hatası: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Doğrulama demo hatası: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Askeri Hava Aracı Tespiti - Demo')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['validation', 'training', 'inference', 'all'],
                       help='Demo modu (varsayılan: all)')
    parser.add_argument('--quick', action='store_true',
                       help='Hızlı eğitim modu (az epoch)')
    
    args = parser.parse_args()
    
    print("🎯 Askeri Hava Aracı Tespiti - Demo Başlatılıyor")
    print("=" * 50)
    
    # Bağımlılıkları kontrol et
    if not check_dependencies():
        print("\n❌ Demo çalıştırılamıyor. Önce gerekli kütüphaneleri yükleyin.")
        return
    
    success_count = 0
    total_demos = 0
    
    # Demo moduna göre çalıştır
    if args.mode in ['validation', 'all']:
        total_demos += 1
        if run_validation_demo():
            success_count += 1
    
    if args.mode in ['training', 'all']:
        total_demos += 1
        if run_training_demo(quick_mode=args.quick):
            success_count += 1
    
    if args.mode in ['inference', 'all']:
        total_demos += 1
        if run_inference_demo():
            success_count += 1
    
    # Sonuç özeti
    print("\n" + "=" * 50)
    print(f"🎊 Demo Tamamlandı: {success_count}/{total_demos} başarılı")
    
    if success_count == total_demos:
        print("✅ Tüm demo'lar başarıyla çalıştı!")
        print("\n📚 Sonraki adımlar:")
        print("1. README.md dosyasını inceleyin")
        print("2. Kendi görüntülerinizle test edin")
        print("3. Model parametrelerini optimize edin")
    else:
        print("⚠️ Bazı demo'lar başarısız oldu. Logları kontrol edin.")

if __name__ == '__main__':
    main()
