from ultralytics import YOLO
import cv2

model = YOLO("my_model.pt")
results = model.predict("IMG_64062.jpg")[0]
img = cv2.imread("IMG_64062.jpg")
boxes = [box.xyxy[0] for box in results.boxes]
boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
#Đổi box
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    radius = max((x2 - x1), (y2 - y1)) // 2
    cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), 2)
    label = f"{i + 1}"
    cv2.putText(img, label, (center_x - 10, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Detection & Count (Sorted)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
