import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    # choose your yaml file
    model = RTDETR('D:\\ultralytics-main\\ultralytics\\cfg\\models\\rt-detr\\rtdetr-x.yaml')
    model.info(detailed=True)
    model.profile(imgsz=[640, 640])
    model.fuse()