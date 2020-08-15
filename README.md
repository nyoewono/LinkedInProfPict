# LinkedInProfPict

## About
This repository consist of 2 main script: main.py and face_recognition.py. The main.py code set an example on how to use the face_recognition class module in order for user 
that does not familiar with code could use the script. This repository mainly is about taking a profile picture from linkedin profile using a specified url given by the user, 
and analyse the image for its sentiment, teeth appearance, and visual recognition using Miscrosoft Computer Vision API. 

## Steps
1. To run the program, one must pip install the packages library that is used in the class script (face_recognition).
2. Create a Microsoft Computer Vision API first and get the subs key and endpoint
3. Create an environment file and fill the following keys:

- USERNAME: .... (linkedin email address)
- PASSWORD: .... (linkedin password)
- COMPUTER_VISION_SUBSCRIPTION_KEY: ....
- COMPUTER_VISION_ENDPOINT: ....

4. Open terminal and locate this folder that you have already downloaded
5. Type "python main.py"
6. Follow the leads on the script output

## Result
The module will output the profile picture from the link. It will also blured the image background from the face. Also, a csv file will be updated (analyzed_images.csv) that list
out the name of the image, indication if there is a teeth or not, caption, category and object tags. 

### Result Detail:
- Show_teeth: 0 or 1 (0 = No teeth detected, 1 = Teeth detected) (int)
- Sentiment: The sentiment of the face (str) (Angry', 'Disgust', 'Fear','Happy', 'Neutral', 'Sad', 'Surprise')
- Caption: The caption for the image (Created by Microsoft Computer Vision API) (str)
- Category: The category for the image (Created by Microsoft Computer Vision API) (str)
- Object tags: List of object that may be recognised by the Microsoft Computer Vision API (list)

### Prone BUG:
- User may get an error saying something about the googlechrome. If this happen, search google chrome for selenium and download the version that match your current google chrome
version. Replace the google chrome in the driver folder with the downloaded one.
- Warnings may appear about the tensor flow, but this will not effect the result.
- Slow internet connection may disrupt selenium worker on scraping the image.
