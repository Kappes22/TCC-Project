import cv2
import numpy as np
import tensorflow as tf
import base64
import time
from os.path import dirname, join
import zmq
import pickle
import codecs
import zlib

#tf.compat.v1.disable_eager_execution()

class Predictor:
    def __init__(self):
        start_Time = time.time()
        cascPath = join(dirname(__file__), "haarcascade_frontalface_default.xml")
        self.faceCascade = cv2.CascadeClassifier(cascPath)
        self.faceDetection = True

        modelPath = join(dirname(__file__), "resNet101v2BeforeOut-2.1.0.tflite")
        self.modelPrunedBeforeOut = tf.lite.Interpreter(model_path=modelPath)
        self.modelPrunedBeforeOut.allocate_tensors()
        self.modelPrunedBeforeOutinput_details = self.modelPrunedBeforeOut.get_input_details()
        self.modelPrunedBeforeOutoutput_details = self.modelPrunedBeforeOut.get_output_details()

        modelPath = join(dirname(__file__), "resNet101v2Before-2.1.0.tflite")
        self.modelPrunedBefore = tf.lite.Interpreter(model_path=modelPath)
        self.modelPrunedBefore.allocate_tensors()
        self.modelPrunedBeforeinput_details = self.modelPrunedBefore.get_input_details()
        self.modelPrunedBeforeoutput_details = self.modelPrunedBefore.get_output_details()


        modelPath = join(dirname(__file__), "resNet101v2After-2.1.0.tflite")
        self.modelPrunedAfter = tf.lite.Interpreter(model_path=modelPath)
        self.modelPrunedAfter.allocate_tensors()
        self.modelPrunedAfterinput_details = self.modelPrunedAfter.get_input_details()
        self.modelPrunedAfteroutput_details = self.modelPrunedAfter.get_output_details()

        self.xface = 0
        self.yface = 0
        self.wface = 0
        self.hface = 0
        self.img_height = 90
        self.img_width = 90
        self.treat_time= 0
        self.predict_time = 0
        self.hasFace = False
        self.labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral' ,'sad' ,'surprised']
        self.cutoff = 0.6
        self.startup_time = start_Time - time.time()
        self.compression_time = 0
        self.maximum = 0
        self.TemperatureEarly = 1.4793775081634521
        self.TemperatureFinal = 2.6395864486694336

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://kappestcc.ddns.net:5555")

    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def earlyExit(self,x,earlyExitInstruction):
         ##EarlyExit
        start_Time = time.time()

        if(earlyExitInstruction!=5):

         self.modelPrunedBefore.set_tensor(self.modelPrunedBeforeinput_details[0]['index'], x)
         self.modelPrunedBefore.invoke()
         middleTensor = self.modelPrunedBefore.get_tensor(self.modelPrunedBeforeoutput_details[0]['index'])
         self.middleTensor = middleTensor


         self.modelPrunedBeforeOut.set_tensor(self.modelPrunedBeforeOutinput_details[0]['index'], middleTensor)
         self.modelPrunedBeforeOut.invoke()
         output = self.modelPrunedBeforeOut.get_tensor(self.modelPrunedBeforeOutoutput_details[0]['index'])
         prediction = self.softmax(output[0]/self.TemperatureEarly)
         outPred = np.argmax(prediction)
         confidence = prediction[outPred]
         self.predict_time = time.time() - start_Time


         early = " EarlyExit"
         if((float(prediction[np.argmax(prediction)])<self.cutoff and earlyExitInstruction == 0) or earlyExitInstruction == 2):
             self.sendMiddleTensor(middleTensor)
             data = self.receiveData()
             early = " Not EarlyExit"
             self.predict_time = time.time() - start_Time

             outPred = int(data[0])
             confidence = data[1]
         elif((float(prediction[np.argmax(prediction)])<self.cutoff and earlyExitInstruction == 3) or earlyExitInstruction == 4):
             self.modelPrunedAfter.set_tensor(self.modelPrunedAfterinput_details[0]['index'], middleTensor)
             self.modelPrunedAfter.invoke()
             output = self.modelPrunedAfter.get_tensor(self.modelPrunedAfteroutput_details[0]['index'])
             prediction = self.softmax(output[0]/self.TemperatureFinal)
             self.predict_time = time.time() - start_Time

             early = " Not EarlyExit(Phone)"
             outPred = np.argmax(prediction)
             confidence = prediction[outPred]
             self.sendData(2)
             data = self.receiveData()
#         else:

#             self.sendData(1)
#             data = self.receiveData()
        else:
         self.sendData(x)
         data = self.receiveData()
         early = " From complete server"
         outPred = int(data[0])
         confidence = data[1]
         self.predict_time = time.time() - start_Time

        return outPred,early,confidence

    def treatImage(self,data):
        #jpg_original = base64.b64decode(data)
        jpg_as_np = np.asarray(data, dtype=np.uint8)
        #jpg_as_np = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(jpg_as_np, cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.hasFace = False
        if(self.faceDetection):
            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            biggestFace = 0
            for (xface, yface, wface, hface) in faces:
                if(wface+hface>biggestFace):
                    self.xface = xface
                    self.yface = yface
                    self.wface = wface
                    self.hface = hface
                    self.hasFace = True
                    crop_img = gray[yface:yface+hface, xface:xface+wface]
                    biggestFace=hface+wface




        textoutput = ""
        if(self.hasFace):
            treated_img = cv2.resize(crop_img, (self.img_height, self.img_width))
            textoutput = "Has a face with the emotion : "
        else:
            self.xface = 73
            self.yface = 115
            self.wface = 136
            self.hface = 136
            crop_img = gray[self.yface:self.yface+self.hface, self.xface:self.xface+self.wface]
            treated_img = cv2.resize(crop_img, (self.img_height, self.img_width))
            textoutput = "No face found, the emotion is : "

        return treated_img.reshape((1,self.img_height,self.img_width,1)).astype("float32"),textoutput

    def predict(self,data,earlyExitInstruction):
        start_Time = time.time()
        x,textoutput = self.treatImage(data)

        self.treat_time = time.time() - start_Time
        prediction,earlyText,confidence = self.earlyExit(x,0)

        TimeText = " discovered face in : "+str(self.treat_time)+ "s" + "s"+" predicted the image in : "+str(self.predict_time)+ "s"

        return self.labels[prediction]+TimeText+earlyText

    def chunks(self,lst, n):
            """Yield successive n-sized chunks from lst."""
            vector = []
            for i in range(0, len(lst), n):
                vector = vector+ [lst[i:i + n]]
            return vector

    def receiveData(self):
        message = self.socket.recv()
        obj_reconstituted = pickle.loads(codecs.decode(message, "base64"))
        return obj_reconstituted

    def sendData(self,data):
        self.compression_time = time.time()
        obj_base64string = codecs.encode(pickle.dumps(np.float16(data), protocol=4), "base64")
        self.compression_time = time.time() - self.compression_time
        self.socket.send(obj_base64string)
    def sendMiddleTensor(self,data):
        self.compression_time = time.time()
        data[data > 20] = 20
        data[data < -20] = -20
        self.compression_time = time.time() - self.compression_time
        self.socket.send(np.int8((data)*6.4))

predictor = Predictor()
def init():
    return "Please select an image!"

def main(data,earlyExitInstruction):
    return predictor.predict(data,earlyExitInstruction)

def getxFace():
    return predictor.xface

def getyFace():
    return predictor.yface

def getwFace():
    return predictor.wface

def gethFace():
    return predictor.hface

def setFaceDetection(faceDetection):
    predictor.faceDetection = faceDetection
