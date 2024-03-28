import cv2 as cv
import time

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
RED = (0, 0, 255)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)
fonts = cv.FONT_HERSHEY_COMPLEX



class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

def ObjectDetector(image):

    classes, scores, boxes = model.detect(
        image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid], score)
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-10), fonts, 0.5, color, 2)

camera = cv.VideoCapture(0)
counter = 0
capture = False
number = 0

start_time = time.time()

while True:
    ret, frame = camera.read()

    orignal = frame.copy()
    ObjectDetector(frame)
    cv.imshow('oringal', orignal)

    # Odottaa 10 sekuntia 'c'-näppäimen painalluksen jälkeen ennen kuvan ottamista
    if capture and time.time() - start_time >= 10:
        cv.putText(
            frame, f"Capturing Img No: {number}", (30, 30), fonts, 0.6, PINK, 2)
        cv.imshow('frame', frame)

        # Odotetaan hetki kuvan näyttämiseksi käyttäjälle
        cv.waitKey(1000)

        # Tallennetaan kuva ja nollataan laskurit
        cv.imwrite(f'ReferenceImages/17_3_24_image{number}.png', orignal)
        capture = False
        counter = 0

    cv.imshow('frame', frame)
    key = cv.waitKey(1)

    if key == ord('c'):
        capture = True
        number += 1
        start_time = time.time()  # Päivitetään aloitusaika 'c' painettaessa
    elif key == ord('q'):
        break


cv.destroyAllWindows()
camera.release()
