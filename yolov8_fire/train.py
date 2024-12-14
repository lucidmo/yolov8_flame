import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')   #ultralytics/cfg/models/v8/yolov8  #yolov8-C2f-Faster-EMA-slimncek.yaml
    #model = YOLO('D:\\ultralytics-main\\runs\\train\\exp2\\weights\\last.pt')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=10,
                workers=8,
                iou=0.5,
                #device='0',
                optimizer='SGD',  # using SGD
                resume='D:\\ultralytics-main\\runs\\train\\exp2\\weights\\last.pt',  # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )