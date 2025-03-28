# -*- coding: utf-8 -*-

"""Created on Sun Mar 17 17:48:27 2024
@author: Kuru Anssi
This script is for running measurement for one file only."""

import cv2 as cv  # Tuodaan OpenCV-kirjasto cv2

# Määritetään vakiot
# KNOWN_DISTANCE = 300  # cm
# PERSON_WIDTH = 16  # INCHES
Todellinen_korkeus = 177  # Oma pituus
todellinen_leveys = 46  # Suunnilleen hauiksen kohdalta, kädet kiinni vartalossa

# Määritellään kynnysarvot YOLO:lle
CONFIDENCE_THRESHOLD = 0.4 #Luvun alittavia rajauslaatikoita ei näytet
NMS_THRESHOLD = 0.3 # Kynnysarvo, jolla estetään päällekkäiset rajauslaatikot

# Määritellään värit
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
RED =(0,0,255)
# Määritetään fontti 
FONTS = cv.FONT_HERSHEY_COMPLEX

# Haetaan luokkanimet classes.txt tiedostosta 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Määritetään YOLO-verkko OpenCV:lle
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# Asetetaan YOLO-verkko käyttämään CUDAa nopeutukseen
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

# Luodaan mallin käsittelyobjekti
model = cv.dnn_DetectionModel(yoloNet)
# Asetetaan syöttöparametrit: koko 416x416, skaalakerroin, RGB-kanavien vaihto
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# TÄSSÄ ON OBJECT DETECTOR -FUNKTIO
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # Luodaan lista, johon YOLO voi palauttaa tiedot
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        # Jokainen luokka saa kehyksen värin clasid-numeron mukaan 
        color= COLORS[int(classid) % len(COLORS)]
    
        # Ilmoitetaan tunnistettu objekti ja sen varmuusaste prosentteina
        label = "%s : %.2f (varmuusaste)" % (class_names[classid], score * 100)
        
        # print(f"tämä box{box}") Tämä oli muuttujien monitorointii kpl 6.4.4
        # print(f"tämä boxes{boxes}") Tämä oli muuttujien monitorointii kpl 6.4.4

        #Piirretään rajauslaatikko kuvaan ja lisätään luokan nimi (label)
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        # Haetaan tiedot ja lisätään ne data_list-listaan
        # 1: Luokka  2: objektin leveys, 3: sijainti, johon korkeustieto kirjoitetaan.
        # LISÄTTY box[3] joka on RAJAUSLAATIKON KORKEUS.
        if classid ==0: # person class id 
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2), box[3]])
        # JOS HALUTAAN LISÄÄ OBJEKTEJA LISÄTÄÄN ELIF- LAUSEKKEITA TÄNNE

    return data_list # Palautetaan listaan lisätyt tiedot joita käsitellä MAIN:issa

# Polttovälin laskeminen
def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# Etäisyyden laskeminen
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance

# Korkeuden mittaaminen
def korkeus_finder(todellinen_korkeus, ref_objectin_korkeus_pix, korkeus_in_frame_pix):
    korkeus = (korkeus_in_frame_pix / ref_objectin_korkeus_pix) * todellinen_korkeus
    return korkeus

# Leveyden mittaaminen
def leveys_finder(todellinen_leveys, ref_objectin_leveys_pix, leveys_in_frame_pix):
    leveys = (leveys_in_frame_pix / ref_objectin_leveys_pix) * todellinen_leveys
    return leveys

# Luetaan referenssikuva
ref_person = cv.imread('ReferenceImages/image1.png')

# Haetaan referenssitiedot
person_data = object_detector(ref_person)  # Kutsutaan object_detector-funktiota
#person_width_in_rf = person_data[0][1]
ref_objectin_korkeus_pix = person_data[0][3]
ref_objectin_leveys_pix = person_data[0][1]  # Sama kuin person_width mutta haluttiin tehdä uusi muuttuja

# Tulostettu debuggausvaiheessa
#print(f"Referenssin leveys pikseleinä: {person_width_in_rf} Referenssin korkeus pikseleinä: {ref_objectin_korkeus_pix}")

# Lasketaan polttoväli henkilön avulla
#focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)


#### MAIN LOOP #####
#cap = cv.VideoCapture(0) # Jäänne reaaliaikamittauksesta.
#Kameraa ei tarvita kuvatiedoston mittaamisessa.

# Alustetaan lastframe. Tätä käytetään vain reaaliaikamittauksessa
last_frame = None

# Aloitusajan tallentaminen. Vain reaaliaiakmittauksessa
# start_time = time.time()
wait_for_t = False  # Alustetaan muuttuja

image_path = "ReferenceImages/image4.png" # Annetaan kuvatiedosto, jota halutaan mitata
frame = cv.imread(image_path) # Tallennetaan kuva frame-nimiseen taulukkoon. 480x640x3 tensori


data = object_detector(frame) # Kutsutaan Object Detector -funktiota
for d in data:
    if d[0] =='person':
        # distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
        # pituus = distance * 2.54
        x, y = d[2] #Rajauslaatikon koordinaatit korkeus-/leveystietojen tulostusta varten 
            
        # Lisätään tähän korkeuden ja leveyden laskenta        
        korkeus = korkeus_finder(Todellinen_korkeus, ref_objectin_korkeus_pix, d[3])
        leveys = leveys_finder(todellinen_leveys, ref_objectin_leveys_pix, d[1])            

        # Mittausten tulostukset kuvaan
        # cv.putText(frame, f'Korkeus: {round(korkeus,2)} cm', (x+5,y+13), FONTS, 0.48, BLACK, 2)
        cv.putText(frame, f'person Korkeus: {round(korkeus,2)} cm', (x, y-50), FONTS, 0.48, RED, 2)
        cv.putText(frame, f'person Leveys: {round(leveys,2)} cm', (x, y-30), FONTS, 0.48, BLACK, 2)        

# Tallennetaan viimeinen ruutu kuvana
cv.imwrite("Mittausten_validointi/viimeinen_ruutu_yolo_733337666.jpg", frame)

cv.imshow('frame', frame) #Näytetään Kuva
cv.waitKey(0) # Voidaan säätää viive millisekunteina. Turha yksittäisen kuvan mittaamisessa
cv.destroyAllWindows() # Suljetaan kaikki ikkunat