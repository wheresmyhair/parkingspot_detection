from ultralytics import YOLO
from PIL import Image
import utils
import config as cfg

# Load a model
# model = YOLO("yolov8m.yaml")  # build a new model from scratch
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

img = Image.open("./8.jfif")
results = model.predict(source=img, save=True, conf=0.03, iou=0.35, hide_labels=True)

# draw global grid on image 
path_img = './4.jpg'
path_img_out = './4_grid.jpg'


img = Image.open(path_img)
utils.draw_horizon(img, 25)
utils.draw_vertical(img, 25)
img.save(path_img_out)


# draw parking spots on image
img = Image.open('./1.jpg')

SPOTS = cfg.PARKING_SPOTS

for spot in SPOTS:
    utils.draw_bbox(img, SPOTS[spot][0], SPOTS[spot][1], SPOTS[spot][2], SPOTS[spot][3], str(spot))

img.save('./spots.jpg')