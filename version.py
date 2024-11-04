import torch
from ultralytics import YOLO

# Cargar el modelo desde el archivo .pt
model = YOLO('prototipo2.pt')

# Verificar el contenido del modelo
print(model)

# Ejemplo de uso: hacer una predicción con el modelo cargado
# Supongamos que tienes una imagen llamada 'imagen.jpg'
results = model('imagen.jpg')

# Mostrar los resultados
results.show()
