#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:33:59 2020

@author: nathanaelyoewono

The script below is made as an example on how to use the LinkedInProfPict class.
This was made as AI Australia submission for intern position as a Data Scientist.
"""

from face_recognition import LinkedInProfPict
import os

print('Welcome to linked profile visual analysis\n')
print('Before we start, please make sure to set your email address and password in env to log in into linkedin')
print('Also, set your Azure subscription key and endpoint to enable Microsoft Visual API\n')


run = LinkedInProfPict()

# login into linkedin first
run.login()

print('We are all set!\n')
cur_dir = os.getcwd()

files = os.listdir(cur_dir+"/downloaded_images")

# typical for mac user
try:
    files.remove('.DS_Store')
except:
    pass

# setting up for the count file name
if len(files) == 0:
    count = 0
else:
    count = int(files[-1][13])

# loop to run the program
find = True

while(find):
    
    url = input('Please type in the linkedin url: ')
    count += 1
    run.get_profile_pict(url, count)
    run.analyze_face()
    
    print('\n')
    again = input('Do you still want to get another linked profile? (y/n)')
    if again == 'n':
        # logout from linkedin
        run.logout()
        find = False


