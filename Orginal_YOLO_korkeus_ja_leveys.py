# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:36:14 2024

@author: kurua
Remake Khans measurement by Anssi Kuru
"""

import cv2 as cv 
import time


# Distance constants 
KNOWN_DISTANCE = 45 #INCHES
PERSON_WIDTH = 16 #INCHES
Todellinen_korkeus = 177
todellinen_leveys = 46 #suunnilleen hauiksen kohdalta käddet kiinni vartalossa

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
RED =(0,0,255)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color= COLORS[int(classid) % len(COLORS)]
    
#        label = "%s : %f" % (class_names[classid], score)
        label = "%s : %.2f (varmuusaste)" % (class_names[classid], score * 100)
        
        print(f"tämä box{box}")
        print(f"tämä boxes{boxes}")


        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid ==0: # person class id 
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2), box[3]]) # Lisätty box[3] jonka pitäisi olla boxin korkeus
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data. 
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

# KORKEUS_MITTAUS 
def korkeus_finder(todellinen_korkeus, ref_objectin_korkeus_pix, korkeus_in_frame_pix):
    korkeus = (korkeus_in_frame_pix / ref_objectin_korkeus_pix) * todellinen_korkeus
    return korkeus

#Leveys_Mittaus
def leveys_finder(todellinen_leveys, ref_objectin_leveys_pix, leveys_in_frame_pix):
    leveys = (leveys_in_frame_pix / ref_objectin_leveys_pix) * todellinen_leveys
    return leveys
    

# reading the reference image from dir 
ref_person = cv.imread('ReferenceImages/17_3_24_image1.png')
#ref_person = cv.imread('ReferenceImages/Khanreferenssit/image14.png')

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]
ref_objectin_korkeus_pix = person_data[0][3]
ref_objectin_leveys_pix = person_data[0][1] # This is same than person_width_in rf but I want to make new variable


print(f"Referenssin leveyspikseleinä : {person_width_in_rf} Referenssin korkeus pixeleinä : {ref_objectin_korkeus_pix}")

# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)


#### MAIN LOOP #####
cap = cv.VideoCapture(0)

# Initialize lst frame what will be saved afterwards.
last_frame = None

# Aloitusajan tallentaminen
#start_time = time.time()
wait_for_t = False  # Alustetaan muuttuja

while True:
    ret, frame = cap.read()

    data = object_detector(frame) 
    for d in data:
        if d[0] =='person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            pituus = distance * 2.54
            print(pituus)
            x, y = d[2]
            
            #Lisätään tähän se korkeusden laskenta
            
            korkeus = korkeus_finder(Todellinen_korkeus, ref_objectin_korkeus_pix, d[3])
            leveys = leveys_finder(todellinen_leveys, ref_objectin_leveys_pix, d[1])            


#       cv.putText(frame, f'Korkeus: {round(korkeus,2)} cm', (x+5,y+13), FONTS, 0.48, BLACK, 2)
        cv.putText(frame, f'person Korkeus: {round(korkeus,2)} cm', (x, y-50), FONTS, 0.88, RED, 2)
        cv.putText(frame, f'person Leveys: {round(leveys,2)} cm', (x, y-30), FONTS, 0.88, BLACK, 2)        
#        cv.putText(frame, f'Dis: {round(distance,2)} inch', (x+5,y+13), FONTS, 0.48, GREEN, 2)
#        cv.putText(frame, f'Dis: {round(float(pituus),2)} cm', (5,50), FONTS, 0.48, BLACK, 2)
#        cv.putText(frame, f'data-alkio: {d} \n tässä korkeus {korkeus} cm', (5,50), FONTS, 0.48, BLACK, 2)
#        cv.putText(frame, f'korkeus {korkeus} cm', (5,50), FONTS, 0.48, BLACK, 2)

        print(f"tämä on d{d}")
        print(f"tämä on d1  {d[1]}")
        
    cv.imshow('frame',frame)
    
    # Tallennetaan viimeinen ruutu
    last_frame = frame.copy()
    
    key = cv.waitKey(100) # tässä voidaan säätää viivettä
    
    if key == ord('t'):
        wait_for_t = True  # Asetetaan odotustila päälle, jos 't' painetaan
        start_time = time.time()  # Asetetaan aloitusaika
        
    if wait_for_t and time.time() - start_time > 10:  # Jos 't' on painettu ja odotettu 10 sekuntia
        break  # Lopetetaan ohjelma
    
    # 'q':n painaminen lopettaa välittömästi
    if key == ord('q'):
        print(f'ssss:{person_width_in_rf}')
        break
    
    # Tallennetaan viimeinen ruutu kuvana
if last_frame is not None and wait_for_t is True:
    cv.imwrite("Mittausten_validointi/viimeinen_ruutu_yolo_jejejeje.jpg", last_frame)
    
cv.destroyAllWindows()
cap.release()
