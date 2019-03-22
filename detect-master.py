from flask import Flask, render_template
from flask_socketio import SocketIO
import json
import base64
import datetime
import numpy as np
import cv2
import os
import sys
import time
from src.ViolenceDetector import *
import settings.DeploySettings as deploySettings
import settings.DataSettings as dataSettings
import src.data.ImageUtils as ImageUtils
import operator
import random
import glob
import os.path
from processor import process_image
from keras.models import load_model
from PIL import Image
from io import BytesIO


app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)

violenceDetector = ViolenceDetector()

@app.route('/')
def sessions():
   return render_template('index_webcam.html')

def messageReceived(methods=['GET', 'POST']):
   print('message was received!!!')

@socketio.on('DetectFrames')
def handle_my_custom_event(frames, methods=['GET', 'POST']):
   dataFrames=list()
   dataJson = json.loads(str(frames).replace('\'','\"'))
   for item in dataJson['data']:
      imgdata = base64.b64decode(item)
      dataFrames.append(imgdata)
      now = datetime.datetime.now()
      response={'isDone':'false','message':'Cevap zamani:'+str(now.isoformat())}
      socketio.emit('DetectFramesResponse', response, callback=messageReceived)
   print("Eleman sayisi:"+str(len(dataFrames)))
   now = datetime.datetime.now()
   response={'isDone':'true','message':'Cevap zamani:'+str(now.isoformat())}
   socketio.emit('DetectFramesResponse', response, callback=messageReceived)

@socketio.on('Detector')
def Detector(frames, methods=['GET', 'POST']):
   dataJson = json.loads(str(frames).replace('\'','\"'))
   for item in dataJson['data']:
      netInput = ImageUtils.ConvertImageFrom_CV_to_NetInput(readb64(item["img"]))
      isFighting = violenceDetector.Detect(netInput)
      #siddet tespit edildi
      if isFighting:
         response={'isDone':'false','message':'Baslangic:'+str(item['time'])}
         socketio.emit('Detector', response, callback=messageReceived)
      else:
         response={'isDone':'false','message':'Bitis:'+str(item['time'])}
         socketio.emit('Detector', response, callback=messageReceived)
   response={'isDone':'true','message':'tespit bitti'}
   socketio.emit('Detector', response, callback=messageReceived)

def readb64(base64_string):
   cleanData = str(base64_string)[len("data:image/jpeg;base64,"):]
   imgdata = base64.b64decode(cleanData)
   image = Image.open(BytesIO(imgdata))
   return np.array(image)

if __name__ == '__main__':
   socketio.run(app, debug=True)