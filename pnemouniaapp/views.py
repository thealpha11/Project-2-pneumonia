from django.http import HttpResponse 
from django.shortcuts import render, redirect
from .models import radio
import cv2
import os
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow import Graph
def home(request):
    return render(request,"home.html")
def test(request):
   


    


    
    
    xray1=request.FILES['xray']
    

    name=xray1.name
    obj=radio(xray=xray1)
    try:
        obj.save()

    except:
        model = load_model('F:\pnemounia\chest_xray\Resnet50.h5')
        p='media/images/'
    
    
    





        img=image.load_img(os.path.join(p,name),grayscale=True ,target_size=(180,180))
        x=image.img_to_array(img)
        x=x*1/255
        x=x.reshape(1,180,180,1)
            # with model_graph.as_default():
            #     tf_session=tf.compat.v1.keras.backend.get_session()
            #     with tf_session.as_default():
        print(type(x))
        print("this is x",x.shape)
        pred=model.predict(x)
        if pred[0][0]<0.5:
            predictedlabel='Normal'
        else:
            predictedlabel='Pneumonia positive'
       
        l1=[]
        l1.append(predictedlabel)
        l1.append(name)
      
        model = load_model('F:\pnemounia\chest_xray\inceptionv31.h5')
        p='media/images/'
    
    
    





        img=image.load_img(os.path.join(p,name),grayscale=True ,target_size=(180,180))
        x=image.img_to_array(img)
        x=x*1/255
        x=x.reshape(1,180,180,1)
            # with model_graph.as_default():
            #     tf_session=tf.compat.v1.keras.backend.get_session()
            #     with tf_session.as_default():
        print(type(x))
        print("this is x",x.shape)
        pred=model.predict(x)
        if pred[0][0]<0.5:
            predictedlabel='Normal'
        else:
            predictedlabel='Pneumonia positive'
       
       
        l1.append(predictedlabel)
        
        l1.append(0.5058506727218628)
        l1.append(0.7467948794364929)
        l1.append(0.727450966835022)
        l1.append(0.9512820243835449)

        l1.append(0.35891273617744446)
        l1.append(0.8381410241127014)
        l1.append(0.8161925673484802)
        l1.append(0.9564102292060852)

      





        return render(request,"show.html",context={'result':l1})
    
                        
    
    
