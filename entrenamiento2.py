if __name__ == "__main__":
    from ultralytics import YOLO

    # Carga el modelo existente
    model = YOLO("runs/detect/train6/weights/best.pt")

    # Realiza el entrenamiento incremental en GPU (forzando CUDA)
    model.train(
        data='data.yaml',
        epochs=10,
        imgsz=640,
        device='cuda',  # Asegura que 'cuda' esté configurado
        amp=True,
        batch=16,
        workers=4  # Ajusta según tus núcleos disponibles
    )

    # Guarda el modelo actualizado
    model.save('3clases.pt')

    # Evalúa el modelo
    metrics = model.val()
    print(metrics)
