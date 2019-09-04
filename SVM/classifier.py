from joblib import dump, load
import sklearn.multiclass
from sklearn.multiclass import OneVsOneClassifier
from sklearn import svm
import os
import numpy as np
import csv
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import linear_kernel
import copy
from PIL import Image
import math
#from sklearn.metrics import confusion_matrix

class Classifier:
    def __init__(self,name,load_model = True,c=1,gamma="scale",kernel="linear"):
        self.model = None
        self.model_file = "model.joblib"
        self.path = "model/"+name+"/"
        self.model_char_size = 15
        if load_model:
            if os.path.isfile(self.path+self.model_file):
                self.model = load(self.path+self.model_file)
                self.model_char_size = load(self.path+"X.joblib").shape[1]

        if not self.model:
            #self.model = OneVsOneClassifier(svm.SVC(C=c,kernel=kernel,gamma=gamma,verbose=True))
            self.model = OneVsOneClassifier(svm.SVC(C=c,kernel=kernel,gamma=gamma))
            self.model.classes_ = None
            X = []
            Y = []
            with open(self.path+name+".csv", 'r') as cs:
                reader = csv.reader(cs)
                for row in reader:
                    X.append(row[:-1])
                    Y.append(row[-1])

            X = np.array(X).astype('int')
            X = np.where(X==255,1,X)
            Y = np.array(Y)
            self.len_x = len(X)
            self.len_y = len(np.unique(Y))

            dump(X,self.path+"X.joblib")
            dump(Y,self.path+"Y.joblib")

            self.model.fit(X,Y)
            dump(self.model,self.path+self.model_file)

        self.kernel = self.model.get_params()['estimator__kernel']
        self.prev_model = None
        self.X_temp = None
        self.Y_temp = None
        np.set_printoptions(precision=3)

    def prediction_test(self,data,label):
        return str(self.model.predict(data)[0]) == label.rstrip()

    def train(self,data,label):
        info = {}
        #prev dataset
        if os.path.isfile(self.path+'X.joblib'):
            arr = load(self.path+"X.joblib")
        else:
            arr = np.array([])

        if os.path.isfile(self.path+'Y.joblib'):
            arr_y = load(self.path+"Y.joblib")
        else:
            arr_y = np.array([])

        if arr.size == 0:
            arr = data
        else:
            arr = np.concatenate((arr,data), axis=0)

        arr_y = np.append(arr_y,label)

        self.X_temp = arr
        self.Y_temp = arr_y

        if self.model.classes_ is not None:
            info["classes_before"] = self.model.classes_
            info["classes_len_before"] = len(self.model.classes_)
            info["count_model_before"] = len(self.model.estimators_)


        self.prev_model = copy.copy(self.model)
        self.model.fit(arr,arr_y)
        info["classes_after"] = self.model.classes_
        info["classes_len"] = len(self.model.classes_)
        info["count_model"] = len(self.model.estimators_)


        return info

    def save_model(self):
        if self.X_temp is not None and self.Y_temp is not None:
            dtset = np.column_stack((self.X_temp,self.Y_temp))
            with open(self.path+"dataset1.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(dtset)
            dump(self.X_temp,self.path+"X.joblib")
            dump(self.Y_temp,self.path+"Y.joblib")
            dump(self.model, self.path+self.model_file)
        self.X_temp = None
        self.Y_temp = None

    def rollback(self):
        self.model = self.prev_model
        self.X_temp = None
        self.Y_temp = None

    def prediction(self,data,verbose=False):
        X = np.array([data]).astype("int")
        indices = self.model.pairwise_indices_
        pjg = len(self.model.estimators_)

        if indices is None:
             Xs = [X] * pjg
        else:
             Xs = [X[:, idx] for idx in indices]

        predictions = np.vstack([est.predict(Xi)
                               for est, Xi in zip(self.model.estimators_, Xs)]).T

        confidences = np.vstack([self.predict_binary(est, Xi)
                                   for est, Xi in zip(self.model.estimators_, Xs)]).T
        Y = self.votes_count(predictions,confidences)
        info = {}
        info["prediction"] = self.model.classes_[Y["sum_conf"].argmax()]
        if verbose:
            info["votes"] = Y['votes']
            info["n_model"] = pjg
            info["model_ex"] = {}
            info["model_ex"]["negative_class"] = self.model.classes_[0]
            est = self.model.estimators_[Y["sum_conf"].argmax()-1]
            info["model_ex"]["no"] = Y["sum_conf"].argmax()-1
            info["model_ex"]["bias"] = est.intercept_
            info["model_ex"]["df"] = est.decision_function(X)
            info["model_ex"]["n_support"] = est.n_support_
            info["model_ex"]["kernel_type"] = est.get_params()['kernel']
            if info["model_ex"]["kernel_type"] == "linear":
                info["model_ex"]["kernel"] = linear_kernel(est.support_vectors_,X)
                info["model_ex"]["w"] = np.around(est.coef_,2)

            else:
                info["model_ex"]["gamma"] = est._gamma
                info["model_ex"]["kernel"] = rbf_kernel(est.support_vectors_,X,gamma=est._gamma)
                info["model_ex"]["w"] = est.dual_coef_

        return info

    def prediction_bulk(self,X):
        return "".join(self.model.predict(X))

    def predict_binary(self,estimator, X):
        """Make predictions using a single binary estimator."""
        return sklearn.multiclass._predict_binary(estimator,X)

    def votes_count(self,pred, conf):
        n_samples = pred.shape[0]
        info = {}

        n_classes = len(self.model.classes_)
        votes = np.zeros((n_samples, n_classes))
        sum_of_confidences = np.zeros((n_samples, n_classes))

        k = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                sum_of_confidences[:, i] -= conf[:, k]
                sum_of_confidences[:, j] += conf[:, k]
                votes[pred[:, k] == 0, i] += 1
                votes[pred[:, k] == 1, j] += 1

                k += 1
        transformed_confidences = (sum_of_confidences /
                               (3 * (np.abs(sum_of_confidences) + 1)))

        info["votes"] = votes
        info["sum_conf"] = votes+transformed_confidences
        return info

    def accuracy(self,filename,answ = None):
        score = 0
        score1 = 0
        if answ:
            filename = filename.replace("\n","").replace(" ","")
            answ = answ.replace("\n","").replace(" ","")

            #f = list(filename)
            #an = list(answ)

            #print(confusion_matrix(f, an, labels=self.model.classes_))

            for i in range(0,max(len(filename),len(answ))):
                if i >= len(filename) or i >= len(answ):
                    break

                if filename[i].lower() == answ[i].lower():
                    score1+=1
                if filename[i] == answ[i]:
                    score += 1
            count = len(answ)

        else:
            test = open("static/image/character_test/"+filename,'r')
            lines = test.readlines()
            count = len(lines)
            ch = int(math.sqrt(self.model_char_size))
            rs = self.model_char_size
            lst = {}
            for i in lines:
                a = i.split(",")
                data = np.array(a[0:-1]).astype('uint8')
                label = a[-1].rstrip()
                im = [data]
                ori_size = int(math.sqrt(len(data)))

                if len(data) != rs:
                    im = np.array(Image.fromarray(data.reshape((ori_size,ori_size))).resize((ch,ch))).reshape(rs).reshape(1,-1)

                asd = self.model.predict(im)

                if str(asd[0]).lower() == label.lower():
                    score1+=1
                if str(asd[0]) == label:
                    score += 1

        return {"sensitive":score/count*100,"insensitive":score1/count*100}
