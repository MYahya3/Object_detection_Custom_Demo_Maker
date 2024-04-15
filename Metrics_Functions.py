import cv2
from helper_functions import DrawOpacBox

def fontScaFinder(metric_dict, bbox, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=1):
    # Bounding Box Coordinates
    rect_x, rect_y, rect_width, rect_height = bbox
    # Get Font Size relate to BBOX Size
    font_scale = min([(rect_width - 30) / cv2.getTextSize(f"{key}: {value:.2f}", font, 1.0, thickness)[0][0] for key, value in metric_dict.items()])
    if font_scale > 0.80:
        font_scale = 0.80
    # Calculate the maximum width of the key strings and values
    max_key_width = max([cv2.getTextSize(f"{key}", font, font_scale, thickness)[0][0] for key, value in metric_dict.items()])
    return font_scale, max_key_width

########### Write Text within defined Rectangle BBOX with autoscaled Font ###########
def DrawWithText(image, font_scale, max_key_width,metric_dict, bbox, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=1, text_color=(18,18,18)):
    # Bounding Box Coordinates
    rect_x, rect_y, rect_width, rect_height = bbox
    # Calculate the vertical spacing between rows
    vertical_spacing = (rect_height // len(metric_dict))
    # Starting position
    text_y = rect_y + 10
    met_xy = dict()
    # Iterate through the metric_dict and draw each key-value pair
    for key, value in metric_dict.items():
        key_text = key
        value_text = f"{value:.2f}"
        # Calculate text size and baseline for the current key and value
        key_size, _ = cv2.getTextSize(key_text, font, font_scale, thickness)
        value_size, _ = cv2.getTextSize(value_text, font, font_scale, thickness)

        # Calculate X-coordinates for drawing key and value text
        key_x = rect_x + 15
        value_x = rect_x + max_key_width + int((rect_width - max_key_width) / 2.35)
        # Draw the key on the image at the current position
        cv2.putText(image, f"{key_text}:", (key_x, text_y + key_size[1]), font, font_scale, text_color, thickness, cv2.LINE_AA)
        # Draw the value within the available width and adjusted font scale
        cv2.putText(image, value_text, (value_x, text_y + value_size[1]), font, font_scale, text_color, thickness, cv2.LINE_AA)

        met_xy[key_text] = (key_x, text_y + key_size[1])

        text_y += vertical_spacing  # Adjust vertical position for the next row

    return image, met_xy, value_x

######### SAMPLE EXAMPLE HOW IT WORKS ###########
# def main():
#
#     metric_dict = {"Productivity % ": 0, "Productive Time (min)": 0, "Unproductive Time (min)": 0, "Idle Time (min)": 0, "Down Time (min)": 0,
#                    "# of product in WIP": 0,
#                    "# of product in WT": 0}
#
#     ######### Initialize Video Capture #########
#
#     vid_source = "video.mp4"
#     cap = cv2.VideoCapture(vid_source)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#
#     # Frames Counter
#     count = 0
#
#     while True:
#         txt_clr = (0,0,8)
#         ret, image = cap.read()
#         if not ret:
#             break
#         count += 1
#         ######### To select single ROI on 1st Frame in video #########
#         if count == 1:
#             bbox = cv2.selectROI("Select Roi", image,False)
#             cv2.destroyWindow("Select Roi")
#
#         # Process name on which i applied condition
#         process = "Productive Time (min)"
#         # Draw Transparent BBOX in image on selected ROI
#         image = DrawOpacBox(image, alpha=0.8, bbox=bbox, color=(140,140,140))
#
#         # Get Text inside BBOX with autoscaled font and spacing
#         image, key_xy, value_x, fontScale= DrawWithText(image,bbox=bbox, metric_dict=metric_dict, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=1, text_color=txt_clr)
#         (x, y) = key_xy[process]
#         print(key_xy)
#         # Condition to change color
#         if metric_dict[process] > 10:
#             process_v = metric_dict.get(process)
#             txt_clr = (0,255,0)
#             cv2.putText(image, f'{process} :', (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, txt_clr, 1,
#                     cv2.LINE_AA)
#             cv2.putText(image, f'{process_v}', (value_x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, txt_clr, 1,
#                     cv2.LINE_AA)
#         if count % int(fps) == 0:
#             metric_dict[process] += 1
#
#     # Display the image
#         cv2.imshow("Orignal Image with Aligned and Resized Values", image)
#         key = cv2.waitKey(51)
#         if key == ord("q"):
#             break
#
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()
