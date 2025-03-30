from tkinter import *
from tkinter import ttk
import pymysql
from tkinter import messagebox


def homescreen(self,parent):
    USERS.parent=parent
    Label(text="").pack()
    title=Label(top,text="LEAF GUARD : Advanced Plant Disease Detection Using Deep Learning Techniques",fg='red',
			font=('Times',20, "bold"))
    title.pack()
    Label(text="").pack()
    Label(width=50,font=('arial',10, "bold"),text="One of the important sectors of Indian Economy is Agriculture. Employment to almost 50% of the countries workforce is provided \n by Indian agriculture sector.  India is known to be the world's largest producer of pulses, rice, wheat, spices and spice products. \n Farmer's economic growth depends on the quality of the products that they produce,  which relies on the plant's growth and the yield \n they get. Therefore, in field of agriculture, detection of disease in plants plays an instrumental role. Plants are highly prone \n to diseases that affect the growth of the plant which in turn affects the ecology of the farmer. In order to detect a plant disease \n at very initial stage, use of automatic disease detection technique is advantageous. The symptoms of plant diseases are conspicuous \n in different parts of a plant such as leaves, etc. Manual detection of plant disease using leaf images is a tedious job.\n Hence, it is required to develop computational methods which will make the process of disease detection and classification \n using leaf images automatic.").pack()
    Label(text="").pack()
    USERS.loginbtn=Button(top,text="LOGIN",width=15,height=3,font=('times',15,"bold"),command=self.login)
    USERS.loginbtn.pack()
    Label(text="").pack()
	#regbtn=Button(top,text="Registration",width=15,height=3,font=('arial',12,"bold"),command=self.registration)
	#regbtn.pack()
