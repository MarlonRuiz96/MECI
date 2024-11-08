import cv2
from ultralytics import YOLO
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import sys
import os

def resource_path(relative_path):
    """Obtiene la ruta absoluta del recurso, considerando PyInstaller."""
    try:
        # PyInstaller crea una carpeta temporal y almacena el camino en _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Cargar el modelo entrenado
model_path = resource_path('prototipo.pt')
model = YOLO(model_path)

class_names = model.model.names
print("Clases detectadas:", class_names)

# Definir colores para cada clase (asegurando que coincidan con class_names)
colors = {
    'organico': (0, 255, 0),       # Verde
    'reciclable': (255, 0, 0)      # Azul
}

# Crear la ventana principal de Tkinter
root = tk.Tk()
root.title("MECI")

# Configurar el tamaño mínimo de la ventana
root.minsize(800, 600)

# Estilo de la interfaz
style = ttk.Style()
style.theme_use('clam')  # Puedes cambiar el tema si lo deseas

# Crear un Frame principal
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# Crear un Frame para los controles (botones y etiquetas)
controls_frame = ttk.Frame(main_frame)
controls_frame.pack(side=tk.TOP, fill=tk.X)

# Crear un Frame para mostrar la imagen
image_frame = ttk.Frame(main_frame)
image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Botón para cargar imagen
btn_cargar = ttk.Button(controls_frame, text="Cargar Imagen", command=lambda: threading.Thread(target=cargar_imagen).start())
btn_cargar.pack(side=tk.LEFT, padx=5)

# Etiqueta para mostrar el conteo de objetos detectados
label_conteo = ttk.Label(controls_frame, text="Objetos detectados: 0 orgánicos, 0 reciclables", font=("Helvetica", 12))
label_conteo.pack(side=tk.LEFT, padx=20)

# Label para mostrar la imagen
label_imagen = ttk.Label(image_frame)
label_imagen.pack(expand=True)

# Función para cargar y procesar la imagen
def cargar_imagen():
    # Abrir el cuadro de diálogo para seleccionar una imagen
    file_path = filedialog.askopenfilename(
        filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if file_path:
        # Leer la imagen usando OpenCV
        image = cv2.imread(file_path)

        # Realizar la predicción
        results = model.predict(source=image, save=False, conf=0.25)

        # Inicializar el conteo de objetos
        conteo = {'organico': 0, 'reciclable': 0}

        # Procesar los resultados
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])            # Índice de la clase
                conf = box.conf[0]               # Confianza
                class_name = class_names[cls]    # Nombre de la clase

                # Actualizar el conteo de objetos detectados
                if class_name in conteo:
                    conteo[class_name] += 1

                # Definir color según la clase
                color = colors.get(class_name, (0, 255, 0))  # Verde por defecto

                # Obtener coordenadas de la caja delimitadora
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Dibujar la caja delimitadora
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Preparar el texto a mostrar (nombre de la clase y confianza)
                text = f'{class_name}: {conf:.2f}'

                # Colocar el texto en la imagen
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

        # Actualizar la etiqueta de conteo de objetos
        label_conteo.config(text=f"Objetos detectados: {conteo['organico']} orgánicos, {conteo['reciclable']} reciclables")

        # Convertir la imagen de OpenCV (BGR) a PIL Image (RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image_rgb)

        # Redimensionar la imagen si es muy grande
        img.thumbnail((800, 600))

        imgtk = ImageTk.PhotoImage(image=img)

        # Mostrar la imagen en la interfaz
        label_imagen.imgtk = imgtk
        label_imagen.configure(image=imgtk)
        label_imagen.image = imgtk  # Evitar que la imagen sea recolectada por el recolector de basura

# Ejecutar el bucle principal de Tkinter
root.mainloop()
