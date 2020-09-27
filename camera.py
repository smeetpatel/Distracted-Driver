import cv2
import tensorflow.keras as k
import numpy as np
import imutils
import tensorflow as tf
from django.conf.locale import fr
from mtcnn.mtcnn import MTCNN
from pip._vendor.certifi.__main__ import args


from model import DriverBehaviourModel
# tf.compat.v1.disable_eager_execution()
# model = DriverBehaviourModel( "model.json", "model_weights_ADAM.h5" )
tf.keras.backend.clear_session()

session = tf.compat.v1.Session()
with session.graph.as_default():
    k.backend.set_session( session )
    model = DriverBehaviourModel( "model.json", "model_weights_ADAM.h5" )
# model = DriverBehaviourModel( "model.json", "model_weights_ADAM.h5" )

font = cv2.FONT_HERSHEY_SIMPLEX


class VideoCamera( object ):
    def __init__(self) -> object:
        """

        :rtype: object
        """
        # Arguement '0' takes feed from camera
        self.video = cv2.VideoCapture(0)


    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    @property
    def get_frame(self):

        _, fr = self.video.read()
        # fr = imutils.resize( fr, width=400 )
        #
        # modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
        # configFile = "deploy.txt"
        # net = cv2.dnn.readNetFromCaffe(configFile, modelFile )
        # (h, w) = fr.shape[:2]
        # blob = cv2.dnn.blobFromImage(cv2.resize( fr, (300, 300)), 1.0,
        #                               (300, 300), (104.0, 177.0, 123.0) )
        # net.setInput( blob )
        # detections = net.forward()
        # # loop over the detections
        # for i in range( 0, detections.shape[2] ):
        #     # extract the confidence (i.e., probability) associated with the
        #     # prediction
        #     confidence = detections[0, 0, i, 2]
        #     # filter out weak detections by ensuring the `confidence` is
        #     # greater than the minimum confidence
        #     if confidence < 0.75:
        #         continue
        #     # compute the (x, y)-coordinates of the bounding box for the
        #     # object
        #     box = detections[0, 0, i, 3:7] * np.array( [w, h, w, h] )
        #     (startX, startY, endX, endY) = box.astype( "int" )
        #
        #     # draw the bounding box of the face along with the associated
        #     # probability
        #     text = "{:.2f}%".format( confidence * 100 )
        #     y = startY - 10 if startY - 10 > 10 else startY + 10
        #     cv2.rectangle( fr, (startX, startY), (endX, endY),
        #                    (0, 0, 255), 2 )
        #     cv2.putText( fr, text, (startX, y),
        #                  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2 )
        #
        #


        pixels = np.asarray( fr )
        detector = MTCNN()
        result = detector.detect_faces(pixels )
        if result:
            for person in result:
                bounding_box = person['box']

                cv2.rectangle( fr,
                               (bounding_box[0], bounding_box[1]),
                               (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                               (0, 155, 255), 2 )
            fc = pixels[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]
            print(fc)
            roi = cv2.resize(fc, (224, 224))
            print(roi)
            with session.graph.as_default():
                k.backend.set_session( session )
                pred = model.predict_emotion( roi[np.newaxis, :, :] )
            cv2.putText(fr, pred, (bounding_box[0], bounding_box[1]), font, 2, (0, 0, 255), 3 )
        # # # # cv2.rectangle( fr, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2 )

        _, jpeg = cv2.imencode( '.jpg', fr )
        return jpeg.tobytes()
