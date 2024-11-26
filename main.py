import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader
# load YAML
import onnx

class yolo_pred():
    def __init__(self,onnx_model,data_yaml):
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f,Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load Yolo model
    def predictions(self, image):
        row, col, d = image.shape

        # CONVERT image into square image
        rc_max = max(row, col)
        input_image = np.zeros((rc_max, rc_max, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        # step-2 get predictions

        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (640, 640), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()
        print(preds.shape)

        detections = preds[0]
        boxes = []
        confidences = []
        classes = []
        image_w, image_h = input_image.shape[:2]
        x_fac = image_w / 640
        y_fac = image_h / 640

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.4:
                class_score = row[5:].max()
                class_id = row[5:].argmax()

                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    left = int((cx - 0.5 * w) * x_fac)
                    top = int((cy - 0.5 * h) * y_fac)
                    width = int(w * x_fac)
                    height = int(h * y_fac)

                    box = np.array([left, top, width, height])
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        boxes_list = np.array(boxes).tolist()
        print(boxes_list)
        confidences_list = np.array(confidences).tolist()
        print(confidences_list)
        file = open("Bboxlist","w+")
        index = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.25, 0.45).flatten()
        # print(classes[])
        for ind in index:
            x, y, w, h = boxes_list[ind]
            conf = int(confidences_list[ind] * 100)
            classes_id = classes[ind]
            class_name = self.labels[classes_id]
            file.write(f'{class_name,x,y,w,h}'+"\n")
            text = f'{class_name}: {conf}%'
            print(class_name, conf, x, y, w, h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 255, 0), -1)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 255), 1)
        return image



yolo = yolo_pred('/home/arun/Downloads/Model2-20230829T105211Z-001/Model2/weights/best.onnx', "data.yaml")

cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        print('unable to read')
        break
    pred_img = yolo.predictions(frame)
    cv2.imshow('yolo', pred_img)
    if cv2.waitKey(1) == 27:
        break

'''
img = cv2.imread("/home/arun/Downloads/yoloproj1/car.jpg")
pred_img = yolo.predictions(img)
cv2.imshow("img", pred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cap.release()
#print(input_image)
'''
