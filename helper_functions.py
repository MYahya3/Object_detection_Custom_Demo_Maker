import cv2
from datetime import datetime

#### Draw Transparent Rectangle BBOX ####
def DrawOpacBox(image, alpha=0.3, bbox = None, color=(255 ,255 ,255)):
    if bbox is not None:
        overlay = image.copy()
        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]) ,color, -1)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image

# To get keys from list of model classes to pass in model during detection
def get_keys_by_values(input_dict, target_values):
    return [key for key, value in input_dict.items() if value in target_values]


# To get prtrained model class idx related to classes you want to detect
def modelClassIdx(model, classes = ['person']):
    """
    This Function is to check if list of classes we give exist for pretrained model or not and return idx for each class
    :param model: model that we initialize YOLOV5
    :param classes: list of classes we want to detect
    :return: classes idx in list of pretrained model classes
    """
    # Make the starting alphabet of names lowercase
    lw_classes = [name[0].lower() + name[1:] for name in classes]
    for i in lw_classes:
        if i not in model.names.values():
            raise NameError(f"Invalid Class Name --> {i}\n Supported class names are --> {model.names.values()}")
    cls_idx = get_keys_by_values(model.names, lw_classes)
    return  cls_idx

def frame_maker(timestamp, fps):
    x = datetime.strptime(timestamp, '%M:%S.%f')
    seconds = x.minute * 60 + x.second + x.microsecond / 1000000
    return int(seconds * fps)

#### Video Writer ####
def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))
    return writer


#### Rescale BBOX if image resized ####
def ScaleBbox(bbox, original_image, new_image):
    original_h, original_w = original_image.shape[:2]
    new_h, new_w = new_image.shape[:2]
    # Get ratio of shape b/w new and orignal image
    width_scale = new_w / original_w
    height_scale = new_h / original_h
    x_min, y_min, x_max, y_max = bbox
    x_min_scaled = int(x_min * width_scale)
    y_min_scaled = int(y_min * height_scale)
    x_max_scaled = int(x_max * width_scale)
    y_max_scaled = int(y_max * height_scale)

    return tuple((x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled))