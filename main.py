from ultralytics import YOLO
from PIL import Image
import utils
import config as cfg


# 模型加载
model = YOLO("./model/yolov8m.pt")  


# 主函数
def main(img, threshold=0.5):
    '''
    ## Description:
        主函数

    ## Args:
        img: PIL.Image
        threshold: float, intersection_over_spot阈值

    ## Returns:
        status: list
    '''
    # 分辨率提升
    img = utils.upscale(img)

    # 预测与预测结果处理
    results = model.predict(source=img, save=False, conf=0.01, iou=0.3, hide_labels=True)
    res_cord = []
    res_conf = []

    for result in results:
        res_cord.append(result.boxes.xywhn)
        res_conf.append(result.boxes.conf)

    res_cord = res_cord[0].cpu()
    res_conf = res_conf[0].cpu()

    # 车位状态
    SPOTS = cfg.PARKING_SPOTS
    status = [False]*len(SPOTS)

    for idx_spot, (x_spot, y_spot, w_spot, h_spot) in SPOTS.items():
        x, y, w, h = utils.ConvertCord.normalize(x_spot, y_spot, w_spot, h_spot, cfg.IMAGE_SIZE)
        spot = [x, y, w, h]

        for _, (x_yolo, y_yolo, w_yolo, h_yolo) in enumerate(res_cord):
            box = [x_yolo.item(), y_yolo.item(), w_yolo.item(), h_yolo.item()]
            intersection_over_spot = utils.iobox2(box, spot)
            if intersection_over_spot > threshold:
                status[idx_spot] = True
                break
    
    return status


if __name__ == "__main__":
    img = Image.open("./imgs/4.jpg")
    status = main(img, 0.8)
    print(status)