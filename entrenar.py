if __name__ == "__main__":
    # Importa la librería YOLO de ultralytics
    from ultralytics import YOLO
    import torch

    # Crea un modelo YOLOv8 vacío para entrenar desde cero
    model = YOLO()  # Aquí no cargamos ningún modelo preentrenado

    # Entrena el modelo usando tu dataset y la configuración del archivo data.yaml
    model.train(data='data.yaml', epochs=50, imgsz=640, device='cuda', amp=False)

    # Guarda el modelo entrenado en formato .pt
    model.save('prototipo1.pt')

    # Verifica el desempeño del modelo entrenado (opcional)
    metrics = model.val()
    print(metrics)
