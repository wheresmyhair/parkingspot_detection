import xml.etree.ElementTree as ET
import utils
import config as cfg

def extract_coordinates(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    coordinates = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        coordinates.append((xmin, ymin, xmax, ymax))

    return coordinates

labels = extract_coordinates('./spots/spots.xml')

SPOTS = cfg.PARKING_SPOTS
IMGSIZE = cfg.IMAGE_SIZE

for idx, label in enumerate(labels):
    SPOTS[idx] = list(utils.ConvertCord.cornerxyxy(label[0], label[1], label[2], label[3], 'cornerxywh'))

print(SPOTS)