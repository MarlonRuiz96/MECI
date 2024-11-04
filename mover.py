#Para mover archivos de una carpeta a otra
import os
import shutil

# Definir las rutas principales
modelos_path = r"D:\datasets\modelo"
proyectos_path = r"D:\datasets\proyecto"

# Definir las subcarpetas que se van a mover
subcarpetas = ["test", "train", "valid"]
tipos = ["images", "labels"]  # Carpetas con archivos

# Recorrer las subcarpetas en modelos
for subcarpeta in subcarpetas:
    for tipo in tipos:
        # Crear las rutas para modelos y proyectos
        source_folder = os.path.join(modelos_path, subcarpeta, tipo)
        target_folder = os.path.join(proyectos_path, subcarpeta, tipo)

        # Verificar si las carpetas existen
        if not os.path.exists(source_folder):
            print(f"Carpeta {source_folder} no existe, saltando.")
            continue
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            print(f"Carpeta {target_folder} creada.")

        # Mover los archivos
        for archivo in os.listdir(source_folder):
            source_file = os.path.join(source_folder, archivo)
            target_file = os.path.join(target_folder, archivo)

            if os.path.isfile(source_file):
                shutil.move(source_file, target_file)
                print(f"Moviendo {source_file} a {target_file}")

print("Archivos movidos exitosamente.")