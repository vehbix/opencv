import numpy as np  
import cv2
import matplotlib.pyplot as plt

class Open:
    def __init__(self) -> None:
        pass

    def kose(self):
        resim=cv2.imread("Source/Resim/kose_bulma.jpg")
        griton=cv2.cvtColor(resim,cv2.COLOR_BGR2GRAY)
        griton=np.float32(griton)
        koseler=cv2.goodFeaturesToTrack(griton,300,0.01,10)
        koseler=np.int0(koseler)


        for kose in koseler:
            x,y=kose.ravel()
            cv2.circle(resim,(x,y),3,255,-1)

        resim=cv2.pyrUp(resim)
        cv2.imshow("koseler",resim)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def nesneBul(self):
        img_rgb=cv2.imread("Source/Resim/ana_resim.jpg")
        img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)

        nesne=cv2.imread("Source/Resim/template.jpg",0)

        w,h=nesne.shape[::-1]

        res=cv2.matchTemplate(img_gray,nesne,cv2.TM_CCOEFF_NORMED)
        threshold=0.79

        loc=np.where(res>threshold)

        for n in zip(*loc[::-1]):
            cv2.rectangle(img_rgb,n,(n[0]+w,n[1]+h),(0,255,255),2)

        cv2.imshow("bulunan nesneler",img_rgb)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def eslestir(self):
        

        resim_aranacak=cv2.imread("Source/Resim/kucuk_resim.jpg")
        resim_buyuk=cv2.imread("Source/Resim/buyuk_resim.jpg")

        orb=cv2.ORB_create()
        anahtar1,hedef1=orb.detectAndCompute(resim_aranacak,None)
        anahtar2,hedef2=orb.detectAndCompute(resim_buyuk,None)
        burutal_force=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
        eslesmeler=burutal_force.match(hedef1,hedef2)
        eslesmeler=sorted(eslesmeler,key=lambda x:x.distance)
        son_resim=cv2.drawMatches(resim_aranacak,anahtar1,resim_buyuk,anahtar2,eslesmeler[:10],None,flags=2)
        plt.imshow(son_resim)
        plt.show()
    
    def morf1(self):
        img = cv2.imread("Source/Resim/j.png")
        img2 = cv2.imread("Source/Resim/j2.png")
        img3 = cv2.imread("Source/Resim/j3.png")


        kernel=np.ones((3,3),np.uint8)
        erosion=cv2.erode(img2,kernel,iterations=1)
        dilation=cv2.dilate(img3,kernel,iterations=1)

        plt.subplot(321),plt.imshow(img),plt.title("orjinal")
        plt.xticks([]),plt.yticks([])
        plt.subplot(322),plt.imshow(img),plt.title("orjinal")
        plt.xticks([]),plt.yticks([])
        plt.subplot(323),plt.imshow(img2),plt.title("img2")
        plt.xticks([]),plt.yticks([])
        plt.subplot(324),plt.imshow(erosion),plt.title("erosion")
        plt.xticks([]),plt.yticks([])
        plt.subplot(325),plt.imshow(img3),plt.title("img3")
        plt.xticks([]),plt.yticks([])
        plt.subplot(326),plt.imshow(dilation),plt.title("dilation")
        plt.xticks([]),plt.yticks([])

        plt.show()

    def morf2(self):
        kamera=cv2.VideoCapture(0)
        while True:
            ret,frame=kamera.read()
            
            hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            
            dusuk_mavi=np.array([100,60,60])
            ust_mavi=np.array([140,255,255])
            
            
            mask=cv2.inRange(hsv,dusuk_mavi,ust_mavi)
            son_resim=cv2.bitwise_and(frame,frame,mask=mask)
            
            kernel=np.ones((5,5),np.uint8)
            erosion=cv2.erode(mask,kernel,iterations=1)
            delation=cv2.dilate(mask,kernel,iterations=1)
            
            opening=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
            closing=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
            
            cv2.imshow("Video",frame)
            cv2.imshow("mask",mask)
            cv2.imshow("son_resim",son_resim)
            cv2.imshow("erosion",erosion)
            cv2.imshow("delation",delation)
            cv2.imshow("opening",opening)
            cv2.imshow("closing",closing)
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        kamera.release()
        cv2.destroyAllWindows() 

    def aracSay(self):
        backsub=cv2.createBackgroundSubtractorMOG2()
        capture=cv2.VideoCapture("Source/Video/video.avi")
        i=0
        minArea=2600
        while True:
            _,frame=capture.read()
            fgmask=backsub.apply(frame,None,0.02)
            erode=cv2.erode(fgmask,None,iterations=4)
            moments=cv2.moments(erode,True)
            cv2.line(frame,(40,0),(40,176),(255,0,0),2)
            cv2.line(frame,(55,0),(55,176),(255,0,0),2)
            
            cv2.line(frame,(0,50),(320,50),(255,0,0),2)
            cv2.line(frame,(0,65),(320,65),(255,0,0),2)
            
            
            cv2.line(frame,(100,0),(100,176),(0,255,255),2)
            cv2.line(frame,(115,0),(115,176),(0,255,255),2)
            
            cv2.line(frame,(0,105),(320,105),(0,255,255),2)
            cv2.line(frame,(0,130),(320,130),(0,255,255),2)
            
            if moments["m00"]>=minArea:
                x=int(moments["m10"]/moments["m00"])
                y=int(moments["m01"]/moments["m00"])
                
                # print("moment :"+str(moments["m00"])+"x :"+str(x)+" y :"+str(y))
                if(x>40 and x<55 and y>50 and y<65):
                    i+=1
                    print("alt"+str(i))
                elif (x>102 and x<110 and y>105 and y<130):
                    i+=1
                    print("üst"+str(i))
                
            cv2.putText(frame,"Sayi:%r"%i,(200,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow("video",frame)
            key=cv2.waitKey(25)
            if key==ord("q"):
                break
            
        capture.release()
        cv2.destroyAllWindows()    
    
    def arkaplanFiltrele(self):
        cap=cv2.VideoCapture("Source/Video/video2.mp4")
        fgbg=cv2.createBackgroundSubtractorMOG2()

        while True:
            ret,frame=cap.read()
            fgmask=fgbg.apply(frame)
            cv2.imshow("fgmask",fgmask)
            cv2.imshow("orjinal",frame)
            k=cv2.waitKey(25) &0xFF
            if k==ord("q"):
                break
        cap.release()
    
    def bulaniklastir(self):
        kamera=cv2.VideoCapture(0)
        while True:
            ret,goruntu=kamera.read()
            
            hsv=cv2.cvtColor(goruntu,cv2.COLOR_BGR2HSV)
            
        
            dusuk_mavi=np.array([100,60,60])
            ust_mavi=np.array([140,255,255])

            mask=cv2.inRange(hsv,dusuk_mavi,ust_mavi)
            son_resim=cv2.bitwise_and(goruntu,goruntu,mask=mask)
            
            
            kernel=np.ones((15,15),np.float32)/255
            smoothed=cv2.filter2D(son_resim,-1,kernel)
            
            blur=cv2.GaussianBlur(son_resim,(15,15),0)
            
            median=cv2.medianBlur(son_resim,15)
            
            bileteral=cv2.bilateralFilter(son_resim,15,75,75)
            
            
            # cv2.imshow("Video",goruntu)
            # cv2.imshow("mask",mask)
            cv2.imshow("son_resim",son_resim)
            # cv2.imshow("smoothed",smoothed)
            cv2.imshow("blur",blur)
            cv2.imshow("median",median)
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        kamera.release()
        cv2.destroyAllWindows()    

    def cannyKenarTespit(self):
        kamera=cv2.VideoCapture(0)
        while True:
            ret,goruntu=kamera.read()
            
            hsv=cv2.cvtColor(goruntu,cv2.COLOR_BGR2HSV)
            
            dusuk_mavi=np.array([100,60,60])
            ust_mavi=np.array([140,255,255])
            
            laplacian=cv2.Laplacian(goruntu,cv2.CV_64F)
            sobelX=cv2.Sobel(goruntu,cv2.CV_64F,1,0,ksize=5)
            sobelY=cv2.Sobel(goruntu,cv2.CV_64F,0,1,ksize=5)
            

            yazi=cv2.Canny(goruntu,100,120)
            
            mask=cv2.inRange(hsv,dusuk_mavi,ust_mavi)
            son_resim=cv2.bitwise_and(goruntu,goruntu,mask=mask)
                
            cv2.imshow("Video",goruntu)    
            cv2.imshow("yazi",yazi)
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        kamera.release()
        cv2.destroyAllWindows()    

    def cannyResim(self):
        messi="Source/Resim/messi.jpg"
        resim=cv2.imread(messi,0)


        kenarlar=cv2.Canny(resim,100,250)




        plt.subplot(121),plt.imshow(resim,cmap="gray")
        plt.title("orjinal"),plt.xticks([]),plt.yticks([])

        plt.subplot(122),plt.imshow(kenarlar,cmap="gray")
        plt.title("kenarlar"),plt.xticks([]),plt.yticks([])



        plt.show()

    def metin(self):
        img=cv2.imread("Source/Resim/sayfa.jpg")

        griton=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gaus=cv2.adaptiveThreshold(griton,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
        gaus=cv2.pyrUp(gaus)


        cv2.imshow("img",img)
        cv2.imshow("gaus",gaus)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def otsuThresholding(self):
        img = cv2.imread("Source/Resim/gurultuluresim.jpg",0)

        _,th1=cv2.threshold(img,150,255,cv2.THRESH_BINARY)
        _,th2=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        blur=cv2.GaussianBlur(img,(9,9),0)
        _,thblur=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        resimler=[img,0,th1,
                img,0,th2,
                blur,0,thblur]
        basliklar=["orjinal","histogram","basit thresholding",
                "orjinal","histogram","otsu thresholding",
                "guassian blur","histogram","otsu thresholding"]

        for i in range(3):
            plt.subplot(3,3,i*3+1),plt.imshow(resimler[i*3],"gray")
            plt.title(basliklar[i*3]),plt.xticks([]),plt.yticks([])
            plt.subplot(3,3,i*3+2),plt.hist(resimler[i*3].ravel(),256)
            plt.title(basliklar[i*3+1]),plt.xticks([]),plt.yticks([])
            plt.subplot(3,3,i*3+3),plt.imshow(resimler[i*3+2],"gray")
            plt.title(basliklar[i*3+2]),plt.xticks([]),plt.yticks([])



        plt.show()

    def renk(self):
        kamera=cv2.VideoCapture(0)
        while True:
            ret,goruntu=kamera.read()
            
            hsv=cv2.cvtColor(goruntu,cv2.COLOR_BGR2HSV)
            
            dusuk_kirmizi=np.array([150,3,30])
            ust_kirmizi=np.array([190,255,255])
            
            # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
            
            dusuk_mavi=np.array([100,60,60])
            ust_mavi=np.array([140,255,255])
            
            dusuk_beyaz=np.array([0,0,140])
            ust_beyaz=np.array([256,60,256])
            
            dusuk_sarı=np.array([5,100,100])
            ust_sarı=np.array([40,255,255])
            
            mask=cv2.inRange(hsv,dusuk_beyaz,ust_beyaz)
            son_resim=cv2.bitwise_and(goruntu,goruntu,mask=mask)
            
            
            
            cv2.imshow("Video",goruntu)
            cv2.imshow("mask",mask)
            cv2.imshow("son_resim",son_resim)
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        kamera.release()
        cv2.destroyAllWindows()    

    def thresholding2(self):
        resim=cv2.imread("Source/Resim/gradient.jpg")
        ret,thresh1=cv2.threshold(resim,127,255,cv2.THRESH_BINARY)
        ret,thresh2=cv2.threshold(resim,127,255,cv2.THRESH_BINARY_INV)
        ret,thresh3=cv2.threshold(resim,127,255,cv2.THRESH_TRUNC)
        ret,thresh4=cv2.threshold(resim,127,255,cv2.THRESH_TOZERO)
        ret,thresh5=cv2.threshold(resim,127,255,cv2.THRESH_TOZERO_INV)

        basliklar=["orjinal","BINARY","BINARY_INV","TRUNC","TOZERO","TOZERO_INV"]
        resimler=[resim,thresh1,thresh2,thresh3,thresh4,thresh5]

        for i in range(6):
            plt.subplot(2,3,i+1),plt.imshow(resimler[i],"gray")
            plt.title(basliklar[i])
            plt.xticks([]),plt.yticks([])
        plt.show()

m=Open()
while True:
    print(" 1  : köşe tespiti")
    print(" 2  : nesneyi bulma")
    print(" 3  : resim eşleştirme")
    print(" 4  : morf1")
    print(" 5  : morf2")
    print(" 6  : araş sayma")
    print(" 7  : arka plan filtreleme")
    print(" 8  : bulanıklaştırma")
    print(" 9  : kenar respiti")
    print(" 10 : canny resim")
    print(" 11 : metin ")
    print(" 12 : otsu thresholding")
    print(" 13 : renk")
    print(" 14 : thresholding")
    print("Uygulamalardan çıkmak için 'q' tuşuna basınız")
    user_input=int(input("işleminizi seçiniz : "))
    
    if user_input== 1:
        m.kose()
    if user_input== 2:
        m.nesneBul()
    if user_input== 3:
        m.eslestir()
    if user_input== 4:
        m.morf1()
    if user_input== 5:
        m.morf2()
    if user_input== 6:
        m.aracSay()
    if user_input== 7:
        m.arkaplanFiltrele()
    if user_input== 8:
        m.bulaniklastir()
    if user_input== 9:
        m.cannyKenarTespit()
    if user_input== 10:
        m.cannyResim()
    if user_input== 11:
        m.metin()
    if user_input== 12:
        m.otsuThresholding()
    if user_input== 13:
        m.renk()
    if user_input== 14:
        m.thresholding2()


    

