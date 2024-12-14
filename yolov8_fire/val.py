import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:\\ultralytics-main-past\\weights\\bestg.pt')  #runs/train/exp/weights/best.pt)
    model.val(data='dataset/data.yaml',
              split='val',
              imgsz=640,
              batch=32,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )