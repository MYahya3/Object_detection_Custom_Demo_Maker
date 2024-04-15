import cv2
from Metrics_Functions import DrawWithText, DrawOpacBox, fontScaFinder
from helper_functions import frame_maker
import pandas as pd

### CUSTOM DEMO TESTING Script
def main():

    # read Csv
    csv_df = pd.read_csv('test_csv.csv', index_col=None)
    metric_dict = {"Productivity % ": 0, "Productive Time (min)": 0, "Unproductive Time (min)": 0, "Idle Time (min)": 0,
                   "Down Time (min)": 0,
                   "# of product in WIP": 0,
                   "# of product in WT": 0}
    keys = list(metric_dict.keys())

    vid_source = "demo_output.mp4"
    foneFac = cv2.FONT_HERSHEY_SIMPLEX

    cap = cv2.VideoCapture(vid_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
            fontScale, max_key_width = fontScaFinder(metric_dict=metric_dict, bbox=bbox, font=foneFac, thickness=1)
        image = DrawOpacBox(frame, alpha=0.9, bbox=bbox, color=(210,210,210))
        # Get Text inside BBOX with autoscaled font and spacing
        frame, met_xy, value_x = DrawWithText(image,fontScale, max_key_width, bbox=bbox, metric_dict=metric_dict, font=foneFac, thickness=2, text_color=(0,0,0))

        ####################################################################
        for i in range(len(csv_df)):
            row = csv_df.loc[i]
            process, start, stop, wip, wt = row[0], row[1], row[2], row[3], row[4]
            # checks process time and prints them
            start_frame = frame_maker(start, fps)  # convert start time to frames
            stop_frame = frame_maker(stop, fps)

            if count in range(start_frame, stop_frame):
                (x, y) = met_xy[process]
                cv2.putText(frame, f'{process}:', (x, y), foneFac, fontScale, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'{metric_dict[process]:.2f}', (value_x, y), foneFac, fontScale, (0,255,0), 2,
                            cv2.LINE_AA)
                metric_dict[process] += ((1 / fps) / 60)  # mints
                metric_dict[keys[5]] = wip
                metric_dict[keys[6]] = wt
                if process == keys[1]:
                    (x, y) = met_xy[keys[0]]
                    cv2.putText(frame, f'{keys[0]}:', (x, y), foneFac, fontScale, (0,255,0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'{metric_dict[keys[0]]:.2f}', (value_x, y), foneFac, fontScale, (0,255,0),
                                2, cv2.LINE_AA)
                    metric_dict[keys[0]] += (((1 / fps) / 60) / ((total_frames / fps) / 60)) * 100

            else:
                pass

        cv2.imshow("YOLOV5 with TEXT BBOX", frame)
        if cv2.waitKey(2) & 0xFF == 27:
            break


if __name__ == "__main__":
    main()
