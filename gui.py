from tkinter import *
from tkinter import ttk
import pymysql
from tkinter import messagebox,filedialog
from PIL import Image, ImageTk
import numpy as np

import cv2
from svm import SVM

from sklearn.model_selection import train_test_split

import os
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score
import sys


top=None
loginbtn=None
usertxt=pwdtxt=None
label2=leftimg=win=label3=None
plants = []
labels = []
namelabels = []

X_train=Y_train=None
directory=""
filename=""
images=labels=None
trainData= testData=trainLabels=testLabels=None
obj=SVM()

def homescreen():
    top=Tk()
    top.geometry("900x800")
    top.title("LEAF GUARD : Advanced Plant Disease Detection Using Deep Learning Techniques")
    Label(text="").pack()
    title=Label(top,width=100,text="LEAF GUARD : Advanced Plant Disease Detection Using Deep Learning Techniques",fg='red',
			font=('Times',20, "bold"))
    title.pack()
    Label(text="").pack()
    Label(width=200,font=('arial',10, "bold"),text="One of the important sectors of Indian Economy is Agriculture. Employment to almost 50% of the countries workforce is provided \n by Indian agriculture sector.  India is known to be the world's largest producer of pulses, rice, wheat, spices and spice products. \n Farmer's economic growth depends on the quality of the products that they produce,  which relies on the plant's growth and the yield \n they get. Therefore, in field of agriculture, detection of disease in plants plays an instrumental role. Plants are highly prone \n to diseases that affect the growth of the plant which in turn affects the ecology of the farmer. In order to detect a plant disease \n at very initial stage, use of automatic disease detection technique is advantageous. The symptoms of plant diseases are conspicuous \n in different parts of a plant such as leaves, etc. Manual detection of plant disease using leaf images is a tedious job.\n Hence, it is required to develop computational methods which will make the process of disease detection and classification \n using leaf images automatic.").pack()
    Label(text="").pack()
    
    Label(top,text="").pack();
    
    Label(top,text="LOGIN",fg='red',font=('arial',13, "bold")).pack()
    
    Label(text="").pack()


    Label(top,text="User Name",font=('arial',10, "bold")).pack()
    global usertxt
    usertxt=Entry(top)
    usertxt.pack()
    
    Label(top,text="Password",font=('arial',10, "bold")).pack()
    global pwdtxt
    pwdtxt=Entry(top, show='*')
    pwdtxt.pack()
    
    
    Label(top,text="").pack()
    global loginbtn
    loginbtn=Button(top,width=8,height=2,text="Login",font=('arial',13, "bold"),command=loginvalidate)
    loginbtn.pack()
    global win
    win=top

    top.mainloop()

def fileDialog():
    global filename
    filename = filedialog.askopenfilename(initialdir =  "/", title = "Select AN IMAGE", filetype =
    (("jpeg files","*.jpg"),("png files","*.png")) )
    global canv,label2
    print(filename)

    img = ImageTk.PhotoImage(Image.open(filename))  # PIL solution
   

    #image segmentation
    ImageFile=filename
    text = str(ImageFile)
    print ("\n*********************\nImage : " + ImageFile + "\n*********************")
    img = cv2.imread(text)
    
    img = cv2.resize(img ,((int)(img.shape[1]),(int)(img.shape[0])))
    original = img.copy()
    neworiginal = img.copy() 
    cv2.imshow('original',img)

    #color image
    M = np.ones(img.shape, dtype='uint8') *50
    resultImage = cv2.subtract(img, M)
    cv2.imshow("contract Image", resultImage)

    #Gray Scale Image
    grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("grayscale image",grayimage)

    width=img.shape[1]
    height=img.shape[0]
    print(width,'  ',height)

    #edge detection
    edgeDetection = np.array((
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ), dtype="int")
    edges = cv2.filter2D(img, -1, edgeDetection)
    cv2.imshow('Edge detection', edges)


    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 70, 210)
    cv2.imshow('Edges', edges)
    
    global leafimg
    global top
    print('flie =',filename)
    img = ImageTk.PhotoImage(Image.open(filename))  
    leafimg.configure(image=img)

    #label = Label(top, text = filename)
    #label.pack()
    

    predict()

def homewindow():
          
    SVM.namelabels.clear()
    directory="C:/Users/mohan/Downloads/BLACK-PEARL-main (1)/BLACK-PEARL-main/PlantVillage"
        # Read each directory
    for directoryname in os.listdir(directory):

        if directoryname.startswith("."):
            continue
        else:
            SVM.namelabels.append(directoryname)  # appends all directory names as labels

    print(SVM.namelabels)
    global images
    global labels
    global obj
    images,labels= obj.prepareTrainingData(directory)
    #print("Total images: ", len(images))
    #print("Total labels: ", len(labels))

    global trainData
    global testData
    global trainLabels
    global testLabels
    # step2: split training and test data
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(images), np.array(labels), test_size=0.1,
                                                                  random_state=59)
    obj.clf = svm.SVC(C=1000)

    # Train the model using the training sets
    obj.clf.fit(trainData, trainLabels)
       
    global win
    global top
    win.destroy()
    top=Tk()
    top.geometry("900x800")
    top.title("LEAF GUARD : Advanced Plant Disease Detection Using Deep Learning Techniques")
    Label(text="").pack()
    logoutbtn= Button(top, text = "LOGOUT",bg='red',command = logout)
    logoutbtn.pack()
    
    title=Label(top,width=100,text="LEAF GUARD : Advanced Plant Disease Detection Using Deep Learning Techniques",fg='red',
			font=('Times',20, "bold"))
    title.pack()
    Label(text="").pack()
    Label(top,text="----------------------------------------------------------------------------------------------------------").pack()
    Label(top,text="Plant Type").pack()
    """plant_type = ttk.Combobox(top, 
                            values=[
                                    "Potato", 
                                    "Tomato",
                                    "Pepper"
                                    ])
    plant_type.pack()"""

    Label(text="").pack()
    title=Label(top,text="Select Plant Leaf Image").pack()
    browsebtn= Button(top, text = "  Select A Plant  ",command = fileDialog)
    browsebtn.pack()

    label = Label(top, text = "")
    label.pack()
    global leafimg
    global label3
    label2 = Label(top, text = "")
    label2.pack()

    leafimg = Label(top, text = "")
    leafimg.pack()
    label3 = Message(top, text = "")
    label3.pack()
    top.mainloop()


def logout():
    global top
    top.destroy()
    print('closing')
    
def predict():
    global filename
    global obj
    global top
    global leafimg
    global label3
    global testData
    global testLabels
    image = cv2.imread(filename)
        
    predict = obj.clf.predict(obj.getFeatures(image).reshape(1, -1))[0]
    print("predict :",predict)

    predicts = obj.clf.predict(testData)
    print("Model Accuracy :", accuracy_score(testLabels, predicts) *100)
    accuracy_result = 'Model Accuracy :' +  str( accuracy_score(testLabels, predicts) *100 )
    
            
        
    cv2.putText(image, SVM.namelabels[predict], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    cv2.imshow("Plant", image)

    print("FIle=",filename)
    #img = ImageTk.PhotoImage(Image.open(filename))  
    #Label(top,image=img).pack()
        
    

    disease_predict=SVM.namelabels[predict]
    '''con = pymysql.connect("localhost","root","","plantdb" )

    # prepare a cursor object using cursor() method
    cursor = con.cursor()
    # execute SQL query using execute() method.
    cursor.execute("SELECT precaution from diseases where plant_disease='%s'" % (disease_predict))
    rows = cursor.fetchall()'''
    outtext=accuracy_result +' \n'
    f=open("C:/Users/mohan/Downloads/BLACK-PEARL-main (1)/BLACK-PEARL-main/precautions.txt","r")
    lines=f.read().splitlines()
    f.close()
    for line in lines:
        if line.startswith(disease_predict):
            ss=line.split("@@")
            if disease_predict.find("healthy")!=-1:
                outtext +="IT IS A HEALTH LEAF OF :\n"+disease_predict
            else:
                outtext +="INFECTED WITH :"+disease_predict+"\n\nPRECAUTIONS\n\n"+ss[1]
    
    print("outtext=",outtext)
    label3.configure(text=outtext)
    outputwindow(outtext)
    

def outputwindow(foundtext):
    global filename
    win=Toplevel()
    win.title("Main Menu")
    win.geometry("400x400")
    img = ImageTk.PhotoImage(Image.open(filename))  
    l=Label(win,font=('arial',14, "bold"),image=img)
        
    l.grid(row=2,column=2)
    w = Message(win, text=foundtext)
    w.grid(row=5, column=2)
        
    #Label(win,text=foundtext).grid(row=5,column=1,columnspan=5)
    win.mainloop()

    


def loginvalidate():
        f=open('C:/Users/mohan/Downloads/BLACK-PEARL-main (1)/BLACK-PEARL-main/user.txt','r')
        global usertxt,pwdtxt
        lines=f.read().splitlines()
        f.close()
        username=usertxt.get()
        password=pwdtxt.get()
        global loginbtn
        if username==lines[0] and password==lines[1]:
            loginbtn.configure(state=DISABLED)
            homewindow()
        else:
            msg = messagebox.showinfo("Information","User Not Found, Try Again!!!")
            



# function to convert image to features
def getFeatures(image):
    image = cv2.resize(image, (500, 500))
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    return hist.flatten()



    # Step1:  prepare training data
def prepareTrainingData(folderPath):
    global plants,labels

    # Read each directory
    for directoryName in os.listdir(folderPath):

        if directoryName.startswith("."):
            continue

        # Read each image
        for imageName in os.listdir(folderPath + "/" + directoryName):

            if imageName.startswith("."):
                continue

            # read each image
            imagePath = folderPath + "/" + directoryName + "/" + imageName
            image = cv2.imread(imagePath)

            plants.append(getFeatures(image))
            labels.append(SVM.namelabels.index(directoryName))

    return plants, labels


    
	
