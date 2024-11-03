import json
import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from sklearn.model_selection import train_test_split

# Crear carpetas para guardar las imágenes y las máscaras en train, val, y test
os.makedirs('Breast/train/images', exist_ok=True)
os.makedirs('Breast/train/masks', exist_ok=True)
os.makedirs('Breast/val/images', exist_ok=True)
os.makedirs('Breast/val/masks', exist_ok=True)
os.makedirs('Breast/test/images', exist_ok=True)
os.makedirs('Breast/test/masks', exist_ok=True)

def process_annotations(json_file, img_dir, train_csv, val_csv, test_csv):
    # Cargar el archivo JSON de COCO con todas las anotaciones
    coco = COCO(json_file)

    data = []

    # Procesar las anotaciones y generar las máscaras
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        # Crear una máscara vacía con el mismo tamaño que la imagen
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        # Obtener las anotaciones para la imagen
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            if 'segmentation' in ann:
                rle = coco.annToRLE(ann)
                decoded_mask = maskUtils.decode(rle)
                mask = np.maximum(mask, decoded_mask)

        # Convertir la máscara a una imagen binaria (blanco y negro)
        mask = (mask > 0).astype(np.uint8) * 255

        # Guardar las rutas de las imágenes y máscaras
        mask_filename = os.path.splitext(img_info['file_name'])[0] + '.png'
        data.append({
            'ImageId': img_info["file_name"],
            'MaskId': mask_filename,
            'OriginalImagePath': img_path,
            'MaskArray': mask
        })

    # Dividir los datos en train (80%), val (15%), test (5%)
    train_data, test_data = train_test_split(data, test_size=0.05, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1575, random_state=42)  # Aproximadamente 15% del total

    # Función para guardar los datos en carpetas y generar CSV
    def save_data(subset_data, subset_name, csv_file):
        subset_paths = []
        for item in subset_data:
            img_filename = item['ImageId']
            mask_filename = item['MaskId']
            img_src = item['OriginalImagePath']
            mask_array = item['MaskArray']

            # Copiar la imagen y la máscara a la carpeta correspondiente
            new_img_path = f'Breast/{subset_name}/images/{img_filename}'
            new_mask_path = f'Breast/{subset_name}/masks/{mask_filename}'
            shutil.copy(img_src, new_img_path)

            # Guardar la máscara
            mask_img = Image.fromarray(mask_array)
            mask_img.save(new_mask_path)

            # Agregar la información al CSV
            subset_paths.append({
                'ImageId': f'./Breast/{subset_name}/images/{img_filename}',
                'MaskId': f'./Breast/{subset_name}/masks/{mask_filename}'
            })

        # Guardar los datos en un CSV
        df = pd.DataFrame(subset_paths)
        df.to_csv(csv_file, index=False)

    # Guardar los datos de train, val y test
    save_data(train_data, 'train', train_csv)
    save_data(val_data, 'val', val_csv)
    save_data(test_data, 'test', test_csv)

# Procesar las anotaciones de train y generar los CSVs para train, val, y test
process_annotations('train/train.json', 'train', 'Breast/train.csv', 'Breast/val.csv', 'Breast/test.csv')

print("Conversión completa y archivos CSV generados.")
