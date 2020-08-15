# relevant libraries

# web
from selenium import webdriver
import requests as rq
import urllib.request

# internal system
import os
import time
import sys
from decouple import config

# image preprocessing
import cv2
from keras.preprocessing import image
from PIL import Image
from io import BytesIO

# tensor
import tensorflow as tf

# data preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

class LinkedInProfPict:
    
    """
    This class is created to get the profile picture from the given linkedin
    link, and analyze the picture. The process consist of 2 parts, image web 
    scraping using selenium, and analyze the picture locally. The class does 
    not take any arguments, but the link hyperparameter should be passed into 
    one of its method, that is the get_profile_pict. Please refer to main.py
    on the script example on how to utilise this class. The class will return
    the image downloaded, blured profile pict, and also csv file containing
    the result of the analysist.
    """

    face_size_req = 0.5

    def __init__(self):
        self.dic = {}

    def open_browser(self, path):

        # set the chrome options
        chrome_options = webdriver.ChromeOptions()  
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--incognito")

        # open browser
        browser = webdriver.Chrome(
            executable_path=path, options=chrome_options)

        browser.get('https://www.linkedin.com/home')

        return browser

    def login(self):

        self.browser = self.open_browser(os.path.dirname(
            os.path.abspath(__file__))+'/driver/chromedriver')

        # click sign in
        self.browser.find_elements_by_class_name(
            'nav__button-secondary')[0].click()
        time.sleep(2)

        try:
            username = config('EMAIL')
            password = config('PASSWORD')
        except:
             print(
                'Please set your email address and linkedin password first in the environment')
             sys.exit()

        self.browser.find_elements_by_id('username')[0].send_keys(username)
        self.browser.find_elements_by_id('password')[0].send_keys(password)

        time.sleep(2)
        # click login
        self.browser.find_element_by_xpath(
            '//*[@id="app__container"]/main/div[2]/form/div[3]/button').click()
        time.sleep(2)

    def logout(self):

        self.browser.get('https://www.linkedin.com/m/logout/')
        self.browser.close()
        print('Your linkedin has been logged out')

    def get_profile_pict(self, url, count):

        # self.login()
        self.image_name = "profile_pict_"+str(count)+".jpg"
        self.browser.get(url)
        time.sleep(2)
        img = self.browser.find_element_by_xpath(
            '/html/body/div[7]/div[3]/div/div/div/div/div[2]/main/div[1]/section/div[2]/div[1]/div[1]/div/div/img')
        src_img = img.get_attribute("src")

        # save the image
        self.dic['Name'] = self.image_name
        urllib.request.urlretrieve(src_img, os.path.dirname(
            os.path.abspath(__file__))+"/downloaded_images/"+self.image_name)

        # self.logout()

    def analyze_face(self):
        
        self.img = cv2.imread(os.getcwd()+'/downloaded_images/'+self.image_name, 1)

        # convert the image to grayscale
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # set the face cascade (reduce image by 5% until face is found)
        # detect multiscale detect rectangle face co-ordinates
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            self.gray_img, scaleFactor=1.05, minNeighbors=5)
        
        # for checking the face dimension
        continue_analyze = 1
        
        if len(faces)==0:
            print('No face detected')
            continue_analyze = 0
            
        else:
            
            # get the weight and height of the face dimension
            for x, y, w, h in faces:
    
                face_dim = w*h
                pict_dim = self.img.shape[0]*self.img.shape[1]
                # check if the face size does not meet the requirement size
                if w/self.img.shape[0] < FaceRecog.face_size_req and h/self.img.shape[1] < FaceRecog.face_size_req:
                    print('Face dimension too small, please choose another picture')
                    continue_analyze = 0
        
        # only continue the image analyzation if it fits in the dimension criteria
        if continue_analyze:
            self.blur_background(faces)
            self.detect_teeth()
            self.face_sentiment()

            try:
                subscription_key = config('COMPUTER_VISION_SUBSCRIPTION_KEY')
                endpoint = config('COMPUTER_VISION_ENDPOINT')
                self.visual_features(subscription_key, endpoint)
            except:
                print(
                    "Set the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable, VISION API will not be run")
                  
            df = pd.read_csv('analyzed_images.csv')
            df = df.append(self.dic, ignore_index=True)
            df.to_csv(os.getcwd()+"/analyzed_images.csv", index=False)
            
            cv2.imwrite(os.path.dirname(
                os.path.abspath(__file__))+"/blured_images/"+self.image_name, self.frame)

    def blur_background(self, face):

        # blur the background aside from the face
        for x, y, w, h in face:

            self.face_crop = self.img[y:y+h, x:x+w]
            self.frame = cv2.blur(self.img, ksize=(10, 10))
            self.frame[y:y+h, x:x+w] = self.face_crop

            # bw prep for keras
            self.roi_gray = self.gray_img[y:y+h, x:x+w]

    def detect_teeth(self):
        
        # link cascade: https://github.com/HoorayLee/TeethDetector
        teeth_cascade = cv2.CascadeClassifier('cascadefina.xml')
        teeth = teeth_cascade.detectMultiScale(
            self.frame, scaleFactor=1.05, minNeighbors=5)
        
        if len(teeth)==0:
            print('No teeth detected')
            self.dic['Show_teeth'] = 0
        else:
            # uncomment this line to make s square around on the teeth
            # for x, y, w, h in teeth:
            #     if x != 0 and y != 0:
            #         self.frame = cv2.rectangle(
            #             self.frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
            self.dic['Show_teeth'] = 1
            print('Teeth detected')
                

    def face_sentiment(self):

        # this model is achieved from pre-trained model by Priya Dwivedi (shout out to her for making this happen :) )
        # link: https://towardsdatascience.com/face-detection-recognition-and-emotion-detection-in-8-lines-of-code-b2ce32d4d5de
        emotion_model = tf.keras.models.load_model(
            "emotion_detector_models/model_v6_23.hdf5")
        class_label = ['Angry', 'Disgust', 'Fear',
                       'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # remove all tensorflow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        
        self.roi_gray = cv2.resize(
            self.roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = self.roi_gray.astype("float") / 255.0
        roi = image.img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        result = emotion_model.predict(roi)
        print('Face Sentiment: '+str(class_label[np.argmax(result[0])]))
        self.dic['Sentiment'] = str(class_label[np.argmax(result[0])])
        print('\n')

    def visual_features(self, subscription_key, endpoint):

        # Add your Computer Vision subscription key and endpoint to your environment variables.
        # if 'COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
        
        analyze_url = endpoint + "vision/v3.0/analyze"

        # image path locally
        image_path = os.path.dirname(
            os.path.abspath(__file__))+"/downloaded_images/"+self.image_name

        # Read the image into a byte array
        image_data = open(image_path, "rb").read()
        headers = {'Ocp-Apim-Subscription-Key': subscription_key,
                   'Content-Type': 'application/octet-stream'}
        params = {'visualFeatures': 'Categories,Description,Color'}
        response = rq.post(
            analyze_url, headers=headers, params=params, data=image_data)
        response.raise_for_status()

        # The 'analysis' object contains various fields that describe the image. The most
        # relevant caption for the image is obtained from the 'description' property.
        analysis = response.json()
        # print(json.dumps(response.json()))
        image_caption = analysis["description"]["captions"][0]["text"].capitalize(
        )
        image_category = [i['name'] for i in analysis['categories']]
        object_tags = analysis["description"]["tags"]
        print(image_caption)
        print(image_category)
        print(object_tags)
        self.dic['Caption'] = image_caption
        self.dic['Category'] = image_category
        self.dic['Objects'] = object_tags
        
        # Display the image and overlay it with the caption.
        # image = Image.open(BytesIO(requests.get(image_url).content))
        # plt.imshow(image)
        # plt.axis("off")
        # _ = plt.title(image_caption, size="x-large", y=-0.1)
        # plt.show()

   