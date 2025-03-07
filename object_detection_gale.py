
from ultralytics import YOLO, RTDETR
from gale_conv_det import YoloSliced, RTDetrSliced
import cv2


if __name__ == '__main__':
    model = YOLO('yolo11n.pt')
    sliced_model = YoloSliced(model, alpha=0.1)

    # model = RTDETR("rtdetr-l.pt")
    # sliced_model = RTDetrSliced(model)

    image_path = 'test_img.jpg'
    image = cv2.imread(image_path)

    results = model(image)[0]

    sliced_results = sliced_model(image)[0]

