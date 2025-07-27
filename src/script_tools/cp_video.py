import os
import shutil
import re

# Directorios de origen y destino
origen = ".\VISEM-Tracking\\visem-extracted-30s-excluding-selected-20\\visem-extracted-30s-excluding-selected-20"
destino = ".\data\ViSEM_Tracking_extended"

# Expresión regular para coincidir con "*_0_30.mp4"
patron = re.compile(r"^([^_]+)_0_30\.mp4$")

# Crear el destino si no existe
os.makedirs(destino, exist_ok=True)

# Loop through files in source folder
for archivo in os.listdir(origen):
    match = patron.match(archivo)
    if match:
        nombre_base = match.group(1)  # Extract the first part of the name
        carpeta_destino = os.path.join(destino, nombre_base)
        os.makedirs(carpeta_destino, exist_ok=True)  # Create folder if it doesn't exist
        
        # New filename inside the destination folder
        nuevo_nombre = f"{nombre_base}.mp4"
        ruta_destino = os.path.join(carpeta_destino, nuevo_nombre)
        
        # Copy and rename
        shutil.copy2(os.path.join(origen, archivo), ruta_destino)
        print(f"Copied and renamed: {archivo} → {ruta_destino}")