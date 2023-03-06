from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

class ConvertCord:
    '''
    corner xyxy, center xywh, corner xywh
    '''
    @staticmethod
    def centerxywh(x, y, w, h, style):
        if style == "cornerxyxy":
            return int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
        elif style == "cornerxywh":
            return int(x-w/2), int(y-h/2), w, h
        else:
            raise ValueError("style must be cornerxyxy or cornerxywh")
    
    @staticmethod
    def cornerxywh(x, y, w, h, style):
        if style == "cornerxyxy":
            return x, y, int(x+w), int(y+h)
        elif style == "centerxywh":
            return int(x+w/2), int(y+h/2), w, h
        else:
            raise ValueError("style must be cornerxyxy or centerxywh")
    
    @staticmethod
    def cornerxyxy(x1, y1, x2, y2, style):
        assert x1 <= x2 and y1 <= y2
        if style == "centerxywh":
            return int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1
        elif style == "cornerxywh":
            return x1, y1, x2-x1, y2-y1
        else:
            raise ValueError("style must be centerxywh or cornerxywh")
        
    @staticmethod
    def normalize(x, y, w, h, img_size):
        return x/img_size[0], y/img_size[1], w/img_size[0], h/img_size[1]

def cropimg(img, x, y, w, h):
    return img[y:y+h, x:x+w]

def draw_horizon(img, y_dist, color=(255, 0, 0), width=1):
    x1, x2 = 0, img.size[0]
    draw = ImageDraw.Draw(img)
    for y in range(0, img.size[1], y_dist):
        draw.line((x1, y, x2, y), fill=color, width=width)
        draw.text((x1+10, y+5), str(y), fill=(255, 0, 0), font=ImageFont.truetype('./fonts/MSYH.TTC', 15))

def draw_vertical(img, x_dist, color=(0, 0, 255), width=1):
    y1, y2 = 0, img.size[1]
    draw = ImageDraw.Draw(img)
    for x in range(0, img.size[0], x_dist):
        draw.line((x, y1, x, y2), fill=color, width=width)
        draw.text((x+10, y1+5), str(x), fill=(0, 0, 255), font=ImageFont.truetype('./fonts/MSYH.TTC', 9))

def draw_bbox(img, x, y, width, height, text, text_offset=[0,0]):
    draw = ImageDraw.Draw(img)
    draw.rectangle((x, y, x+width, y+height), outline=(255, 0, 0), width=1)
    draw.text((x+text_offset[0], y-20+text_offset[1]), str(text), fill=(255, 0, 0), font=ImageFont.truetype('./fonts/MSYH.TTC', 15))

def denoise(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img

def upscale(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img

def black_white(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    return img

def iou(box1, box2):
    '''
    ## Description
    Calculate the intersection over union of two bounding boxes.

    ## Args
    box1: (x1, y1, w1, h1)
    box2: (x2, y2, w2, h2)

    ## Returns
    iou: float
    '''
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1, y1, x2, y2 = max(x1, x2), max(y1, y2), min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    if x1 >= x2 or y1 >= y2:
        return 0
    else:
        return (x2-x1)*(y2-y1)/(w1*h1+w2*h2-(x2-x1)*(y2-y1))
    
def iobox2(box1, box2):
    '''
    ## Description
    Calculate the intersection of two box over box2.

    ## Args
    box1: (x1, y1, w1, h1)
    box2: (x2, y2, w2, h2)

    ## Returns
    iobox2: float
    '''
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1, y1, x2, y2 = max(x1, x2), max(y1, y2), min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    if x1 >= x2 or y1 >= y2:
        return 0
    else:
        return (x2-x1)*(y2-y1)/(w2*h2)