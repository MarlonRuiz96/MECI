#Para entrenar con ultratitics
from ultralytics import YOLO, checks, hub
checks()

hub.login('1dbec09462b00a64951045efbffdb8ebf3c7c9d70d')

model = YOLO('https://hub.ultralytics.com/models/LITavh1s0RheboRG9LH7')
results = model.train()
