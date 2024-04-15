import cv2
from YoloDetector import yoloModel, detectObject
from Metrics_Functions import DrawWithText, DrawOpacBox, fontScaFinder


### CUSTOM DEMO TESTING Script
def main():
    metric_dict = {"Productivity % ": 0, "Productive Time (min)": 0, "Unproductive Time (min)": 0, "Idle Time (min)": 0,
                   "Down Time (min)": 0,
                   "# of product in WIP": 0,
                   "# of product in WT": 0}

    vid_source = "vid.mp4"
    class_names = ["Person", "Chair"]
    model_name = 'yolov8'
    model_wieghts = "s"
    model_conf = 0.25
    fontFac = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize Model Get model and model Parameters
    model = yoloModel(model_name=model_name,model_weights=model_wieghts, classes=class_names, detection_conf=model_conf)

    cap = cv2.VideoCapture(vid_source)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Frames Counter
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count == 1:
            bbox = cv2.selectROI("Select Roi", frame,False)
            cv2.destroyWindow("Select Roi")
            fontScale, max_key_width = fontScaFinder(metric_dict=metric_dict, bbox=bbox, font=fontFac, thickness=1)

        image = DrawOpacBox(frame, alpha=0.8, bbox=bbox, color=(140,140,140))

        # Get Text inside BBOX with autoscaled font and spacing
        frame, met_xy, value_x = DrawWithText(image,fontScale, max_key_width, bbox=bbox, metric_dict=metric_dict, font=fontFac, thickness=2, text_color=(0,0,0))

        res_detections, cls_names= detectObject(model_name=model_name,model=model, image=frame)
        for person in res_detections:
            x,y,w,h,conf, label = person
            d_class = cls_names[int(label)].capitalize()

            if d_class == class_names[0]:
                cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (50, 205, 0), 2)
                cv2.putText(frame, d_class, (int(x), (int(y) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 2)
            elif d_class == class_names[1]:
                cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (50, 205, 200), 2)
                cv2.putText(frame, d_class, (int(x), (int(y) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)

        cv2.imshow("YOLOV5 with TEXT BBOX", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


if __name__ == "__main__":
    main()
