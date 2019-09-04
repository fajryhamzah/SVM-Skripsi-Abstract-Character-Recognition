from .classifier import Classifier
from .image_processing import ImageProcessing
from .information_categorize import InformationCategorize
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import os
from joblib import dump, load
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image
#import time
import math
import csv

class Main:
    def __init__(self):
        self.model = None

    def set_model(self,name):
        self.model = Classifier(name)
        self.model_img_size = self.model.model_char_size

    def single_train(self,X,Y,X1=None,Y1=None):
        dat = []
        info = {}
        url_img = X
        X = ImageProcessing(X,grayscale_save=True,model_size=self.model_img_size)
        channel = X.bands
        info["prefix"] = X.prefix
        X = X.process_character()
        lbl = np.array([Y])
        info["first_image"] = X
        info["first_image"]["channel"] = channel
        info["first_image"]["vector"] = X["binary_img"].astype("int")
        info["first_image"]["vector_flat"] = X["binary_img"].astype("int").flatten()
        dat.append(info["first_image"]["vector_flat"])
        info["first_image"]["clean"] = url_img
        info["first_image"]["class"] = Y
        if(X1):
            url_img = X1
            X1 = ImageProcessing(X1,grayscale_save=True,model_size=self.model_img_size)
            channel = X1.bands
            X1 = X1.process_character()
            lbl = np.array([Y,Y1])
            info["second_image"] = X1
            info["second_image"]["channel"] = channel
            info["second_image"]["clean"] = url_img
            info["second_image"]["class"] = Y1
            info["second_image"]["vector_flat"] = X1["binary_img"].astype("int").flatten()
            info["prefixx"] = X1.prefix
            dat.append(info["second_image"]["vector_flat"])

        point = np.array(dat).astype("int")

        info["train"] = self.model.train(point,lbl)
        info["ex"] = {}
        info["ex"]["positif"] = self.model.model.classes_[0]
        est = self.model.model.estimators_[",".join(info["train"]["classes_after"]).replace(",","").index(Y)-1]
        info["ex"]["no"] = info["train"]["count_model_before"]
        info["ex"]["type"] = self.model.kernel
        if(self.model.kernel == "rbf"):
            info["ex"]["gamma"] = est._gamma
            info["ex"]["kernel"] = rbf_kernel(est.support_vectors_,[point[0]],gamma=est._gamma)
            info["ex"]["w"] = est.dual_coef_
        else:
            info["ex"]["kernel"] = linear_kernel(est.support_vectors_,[point[0]])
            info["ex"]["w"] = est.coef_

        info["ex"]["bias"] = est.intercept_
        info["ex"]["n_support"] = est.n_support_
        return info

    def bulk_train(self,kernel,c,gamma,name):
        info = {}

        clf = Classifier(name,load_model=False,c=c,gamma=gamma,kernel=kernel)
        info["data"] = clf.len_x
        info["class"] = clf.len_y
        info["name"] = name
        info["kernel"] = kernel

        return info

    def classifier_test(self,img):
        info = {}
        info["image"] = {}
        img = ImageProcessing(img,grayscale_save=True,model_size=self.model_img_size)
        info["prefix"] = img.prefix
        info["channel"] = img.bands
        img = img.process_character()
        info["image"] = img
        info["image"]["vector"] = info["image"]["binary_img"].astype("int")
        info["image"]["vector_flat"] = info["image"]["binary_img"].astype("int").flatten()
        pred = self.model.prediction(info["image"]["vector_flat"],True)
        info['prediction'] = pred["prediction"]
        info["votes"] = pred["votes"].max().astype("int")
        info["jumlah_model"] = pred["n_model"]
        info["ex"] = pred["model_ex"]
        #info["image"]["rbf"] = rbf_kernel()

        return info

    def test(self,img,skip=None,compare=None):
        info = {}
        info["image"] = {}

        if skip:
            img_real = img
            img = os.path.splitext(os.path.basename(img))[0]+"_"
            info = load("static/cache/"+self.model.path.split('/')[1]+"_"+img+"_dat.joblib")
            info["skip"] = skip
            info["img_name"] = img_real

            if not info["categorize"]:
                info["categorize"] = InformationCategorize(info["classifier"]).get_all()
                dump(info,"static/cache/"+self.model.path.split('/')[1]+"_"+img_real+"_dat.joblib")


            if "udah" not in info:
                for i in info["classifier"]:
                    for a in  i:
                        info["full_text"] += a+" "
                    info["full_text"] += "\n"

                if "categorize" not in info:
                    info["categorize"] = InformationCategorize(info["classifier"]).get_all()
                    #write to csv
                    header = ['judul', 'nama', 'nim', 'isi','keyword']
                    isi = []
                    with open("static/csv/"+info["prefix"]+".csv", "w", newline='') as f:
                        writer = csv.writer(f, delimiter='|')
                        writer.writerow(header)
                        #judul
                        judul = ""

                        for i in range(info["categorize"]["judul_start"],info["categorize"]["judul_end"]+1):
                            judul += " ".join(info["classifier"][i])+"\n"

                        isi.append(judul)
                        #nama
                        isi.append(" ".join(info["classifier"][info["categorize"]["nama"]]))
                        #nim
                        isi.append(" ".join(info["classifier"][info["categorize"]["nim"]]))
                        #isi
                        isii = ""

                        for i in range(info["categorize"]["isi_start"],info["categorize"]["isi_end"]+1):
                            isii += " ".join(info["classifier"][i])+"\n"

                        isi.append(isii)
                        #keyword
                        ky = ""

                        for i in range(info["categorize"]["keyword_start"],info["categorize"]["keyword_end"]+1):
                            ky += " ".join(info["classifier"][i])+"\n"

                        isi.append(ky)
                        writer.writerow(isi)

            # karakter = info["karakter"]
            #
            # no_line = 1
            # for i in karakter:
            # #     #kata
            #     kata = []
            #     kta = 1
            #     for k in i:
            #         ch = 1
            #         for d in k:
            #             d = np.array(d)
            #             d = np.where(d==1,255,d).reshape((int(math.sqrt(self.model_img_size)),int(math.sqrt(self.model_img_size))))
            #             #print(d)
            #             dat = Image.fromarray(d.astype('uint8')).convert("L")
            #             dat.save("static/manual/"+str(no_line)+"_"+str(kta)+"_"+str(ch)+".png")
            #             ch+=1
            #         kta+=1
            #     no_line+=1
        else:
            img = ImageProcessing(img,True,model_size=self.model_img_size)
            info["image"]["size"] = img.size
            info["image"]["binary"] = img.bradley_roth(True)
            info["image"]["skew"] = img.skew_corrected()
            info["image"]["lines"] = img.line_segmentation()
            info["image"]["word"] = img.word_segmentation()
            info["prefix"] = img.prefix
            karakter = img.character_segmentation()
            info["karakter"] = karakter
            info["full_text"] = ""
            text = []
            #
            # f= open("guru99.txt","w+")
            # f.write(str(karakter))
            # f.close()
            #
            # #baris
            for i in karakter:
            #     #kata
                kata = []
                for k in i:
            #         #print(k)
                    kt = self.model.prediction_bulk(k).replace("koma",",").replace("dot",".").replace("slash","/")
                    kata.append(kt)
                    info["full_text"] += kt+" "
            #         kata += " "
                 #print(kata)
                info["full_text"] += "\n"
                text.append(kata)
            info["classifier"] = text
            info["categorize"] = InformationCategorize(info["classifier"]).get_all()
            info["udah"] = True
            dump(info,"static/cache/"+self.model.path.split('/')[1]+"_"+img.prefix+"_dat.joblib")
            #write to csv
            header = ['judul', 'nama', 'nim', 'isi','keyword']
            isi = []
            with open("static/csv/"+img.prefix+".csv", "w", newline='') as f:
                writer = csv.writer(f, delimiter='|')
                writer.writerow(header)
                #judul
                judul = ""

                for i in range(info["categorize"]["judul_start"],info["categorize"]["judul_end"]+1):
                    judul += " ".join(info["classifier"][i])+"\n"

                isi.append(judul)
                #nama
                isi.append(" ".join(info["classifier"][info["categorize"]["nama"]]))
                #nim
                isi.append(" ".join(info["classifier"][info["categorize"]["nim"]]))

                #isi
                isii = ""

                for i in range(info["categorize"]["isi_start"],info["categorize"]["isi_end"]+1):
                    isii += " ".join(info["classifier"][i])+"\n"

                isi.append(isii)
                #keyword
                ky = ""

                for i in range(info["categorize"]["keyword_start"],info["categorize"]["keyword_end"]+1):
                    ky += " ".join(info["classifier"][i])+"\n"

                isi.append(ky)
                writer.writerow(isi)

        if compare:
            ans = open("./static/answer/"+compare,"r").read()
            info["accuracy"] = self.check_accuracy(info["full_text"],ans)


        #print(text)
        return info

    def classifier_rollback(self):
        self.model.rollback()

    def classifier_save(self):
        self.model.save_model()

    def model_list(self):
        nm = [name for name in os.listdir("model") if os.path.isdir("model/"+name)]
        nm.sort()
        return nm

    def list_img_uploaded(self):
        nm = [name for name in os.listdir("static/image/abstrak") if os.path.isfile("static/image/abstrak/"+name)]
        nm.sort()
        return nm

    def accuracy_list(self):
        nm =  [name for name in os.listdir("static/image/character_test") if name.endswith(".csv")]
        nm.sort()
        return nm

    def text_list(self):
        nm = [name for name in os.listdir("static/answer") if name.endswith(".txt")]
        nm.sort()
        return nm

    def check_accuracy(self,name, answer = None):
        return self.model.accuracy(name,answer)
