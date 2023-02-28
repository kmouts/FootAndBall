from ultralytics import YOLO
from ultralytics.yolo.v8.detect import DetectionTrainer

# trainer
trainer = DetectionTrainer(overrides={"data": "soccernet_v3.yaml", "imgsz": 1280,
                                      "batch": -1, "workers": 12,
                                      "pretrained": True, "epochs": 100, "patience": 5,
                                      "model": "yolov8n.pt",
                                      "val": True, "device": 0, "name": "yolo8n_custom_1280_Adam",
                                      "rect": True, "verbose": True, "cache": True,
                                      "optimizer": "Adam"
                                      })
trainer.train()
trained_model = trainer.best
