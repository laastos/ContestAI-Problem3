
# TARGETED DOWNLOAD FUNCTIONS FOR GOOGLE COLAB
# Copy specific functions as needed

import os, zipfile, glob
from google.colab import files
from datetime import datetime

def download_models():
    """Download only model files (.pth, .pt, .pkl)"""
    model_files = []
    for ext in ['.pth', '.pt', '.pkl', '.h5']:
        model_files.extend(glob.glob(f"*{ext}"))
        model_files.extend(glob.glob(f"**/*{ext}", recursive=True))

    model_files = list(set(model_files))
    if not model_files:
        print("❌ No model files found!")
        return

    print(f"🤖 Found {len(model_files)} model files:")
    for f in model_files:
        size = os.path.getsize(f) / 1024 / 1024
        print(f"  • {f} ({size:.1f} MB)")

    zip_name = f"models_{datetime.now().strftime('%H%M%S')}.zip"
    with zipfile.ZipFile(zip_name, 'w') as z:
        for f in model_files: z.write(f, f)

    files.download(zip_name)
    print(f"✅ Downloaded: {zip_name}")

def download_images():
    """Download only image files"""
    img_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.gif']:
        img_files.extend(glob.glob(f"*{ext}"))
        img_files.extend(glob.glob(f"**/*{ext}", recursive=True))

    img_files = list(set(img_files))
    if not img_files:
        print("❌ No image files found!")
        return

    print(f"🖼️ Found {len(img_files)} image files")
    zip_name = f"images_{datetime.now().strftime('%H%M%S')}.zip"
    with zipfile.ZipFile(zip_name, 'w') as z:
        for f in img_files: z.write(f, f)

    files.download(zip_name)
    print(f"✅ Downloaded: {zip_name}")

def download_specific(filenames):
    """Download specific files by name"""
    existing = [f for f in filenames if os.path.exists(f)]
    missing = [f for f in filenames if not os.path.exists(f)]

    if missing:
        print(f"⚠️ Missing: {missing}")

    if not existing:
        print("❌ No files to download!")
        return

    zip_name = f"specific_{datetime.now().strftime('%H%M%S')}.zip"
    with zipfile.ZipFile(zip_name, 'w') as z:
        for f in existing: z.write(f, f)

    files.download(zip_name)
    print(f"✅ Downloaded: {zip_name}")

# Usage examples:
# download_models()
# download_images()  
# download_specific(['cvae_final.pth', 'training_loss.png'])
