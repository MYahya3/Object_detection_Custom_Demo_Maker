import torch
import cv2
from helper_functions import modelClassIdx
from ultralytics import YOLO

# To load and initialize model (YOLOV5 / YOLOV8) with desired parameters
def yoloModel(model_name= None, model_weights="s", detection_conf= 0.10, classes=None):
    # Check if classes defined or not
    if classes is None:
        raise  ValueError("Please provide list of classes to model e.g ['person','car]")

    if model_name == "yolov5":
        # Check if the specified model weight is valid
        weights: str = f"yolov5{model_weights}"
        if f"yolov5{model_weights}" in ["yolov5s", "yolov5m", "yolov5l", "yolov5x"]:
            model = torch.hub.load("ultralytics/yolov5", weights, pretrained=True)
        else:
            raise ValueError("Invalid model weights. Supported wiegths names: s, m, l, x")

    elif model_name == "yolov8":
        if f"yolov8{model_weights}" in ["yolov8s", "yolov8m", "yolov8l", "yolov8x"]:
            model = YOLO(f"yolov8{model_weights}")
        else:
            raise ValueError("Invalid model weights. Supported wiegths names: s, m, l, x")
    else:
        raise NameError("Invalid model name. Supported models: yolov5, yolov8")
    # Define model classes and model confidence thresholdd
    model.classes = modelClassIdx(model, classes=classes)
    model.conf = detection_conf
    return model


# Function to get model image and return detections in each frame (x1,y1,x2,y2,confidence, class_idx)
def detectObject(image,model,model_name= None):
    if model_name == "yolov5":
        result = model(image)
        detections = result.xyxy[0]
    elif model_name == "yolov8":
        result = model(image, classes=model.classes, conf= model.conf, verbose=False)
        detections = result[0].boxes.data
    else:
        raise NameError("Provide Model Name For Detection results")
    return detections, model.names

# def main():
#     vid_source = "video.avi"
#     class_names = ["person","Car"]
#     model_name = 'yolov8'
#     model_conf = 0.50
#
#     # Initialize Model Get model and model Parameters
#     model = yoloModel(model_name=model_name, classes=class_names, detection_conf=model_conf)
#     cap = cv2.VideoCapture(vid_source)
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         results, cls_names= detectObject(model=model, image=frame, model_name=model_name)
#         for bbox in results:
#             x, y, w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
#             cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (50, 205, 200), 2)
#             cv2.putText(frame, "bb", (int(x), (int(y) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                             (0, 0, 0), 2)
#
#         cv2.imshow("RGB", frame)
#         if cv2.waitKey(1) & 0xFF == 27:
#             break
#
# if __name__ == "__main__":
#     main()
