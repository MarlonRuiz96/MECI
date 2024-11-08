import cv2
from ultralytics import YOLO
import threading
import tkinter as tk
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

# Obtener los nombres de las clases desde el modelo
class_names = model.model.names
print("Clases detectadas:", class_names)

# Definir colores para cada clase (asegurando que coincidan con class_names)
colors = {
    'organico': (0, 255, 0),       # Verde
    'reciclable': (255, 0, 0)      # Azul
}

# Crear la ventana principal de Tkinter
root = tk.Tk()
root.title("Detección de Objetos: Papel y Baterías")

# Crear un Label en Tkinter para mostrar el video
label_video = tk.Label(root)
label_video.pack()

# Inicializar la cámara
cap = cv2.VideoCapture(0)

def mostrar_video():
    ret, frame = cap.read()
    if ret:
        # Realizar la predicción
        results = model.predict(source=frame, save=False, conf=0.25)

        # Procesar los resultados
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])            # Índice de la clase
                conf = box.conf[0]               # Confianza
                class_name = class_names[cls]    # Nombre de la clase

                # Definir color según la clase
                color = colors.get(class_name, (0, 255, 0))  # Verde por defecto

                # Obtener coordenadas de la caja delimitadora
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Dibujar la caja delimitadora
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Preparar el texto a mostrar (nombre de la clase y confianza)
                text = f'{class_name}: {conf:.2f}'

                # Colocar el texto en la imagen
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

        # Convertir el frame de OpenCV (BGR) a PIL Image (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Actualizar el Label de Tkinter con la nueva imagen
        label_video.imgtk = imgtk
        label_video.configure(image=imgtk)

    # Programar la siguiente actualización
    label_video.after(10, mostrar_video)

# Iniciar la función de mostrar video
mostrar_video()

# Ejecutar el bucle principal de Tkinter
root.mainloop()

# Liberar la cámara al cerrar la ventana
cap.release()
cv2.destroyAllWindows()
