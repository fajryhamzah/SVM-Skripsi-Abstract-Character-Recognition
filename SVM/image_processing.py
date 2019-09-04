import numpy as np
from PIL import Image
from skimage.transform.integral import integral_image
import copy
import os
import math
from scipy.ndimage import interpolation as inter

class ImageProcessing:
    def __init__(self,image,grayscale_save=False,model_size=225):
        self.ori_image = Image.open(image)
        self.prefix = os.path.splitext(os.path.basename(self.ori_image.filename))[0]+"_"
        self.image =  self.ori_image.convert("L")
        self.size = self.image.size
        self.lines = []
        self.words = []
        self.char_size = int(math.sqrt(model_size))
        self.char_reshape = model_size
        self.bands = self.ori_image.getbands()
        self.char_ratio = 1
        self.binary_image = []
        if grayscale_save:
            self.image.save("./static/cache/"+self.prefix+"ori.jpg")

    def process_character(self,dont_skip_binary=True):
        info = {}
        info['ori_size'] = self.size
        if self.image.size != (self.char_size,self.char_size):
            self.image = self.ori_image.resize((self.char_size,self.char_size))
            img = self.image.convert("L")
            img.save("./static/cache/"+self.prefix+"resize.jpg")
            info["vector_resize"] = np.array(self.image).astype("int").tolist()
            self.image = img

        info["vector_asli"] = np.array(self.ori_image).astype("int").tolist()
        info["vector_grayscale"] = np.array(self.image).astype("int").tolist()

        info["current_size"] = self.image.size

        #skip if image already in binary mode (dont_skip_binary = false)
        if dont_skip_binary:
            info["binary"] = self.bradley_roth(True)


        img = self.binary_image.reshape(self.char_reshape).reshape(1,-1)
        info["vector"] = img
        info["binary_img"] = self.binary_image

        return info

    def bradley_roth(self,verbose=False):
        if verbose: info = {}
        arr = np.array(self.image).astype('int64')
        intImg = integral_image(arr)
        s = np.round(arr.shape[1]/8)
        t = 15
        out = np.zeros(arr.shape)

        for i in range(0,arr.shape[1]):
            for j in range(0,arr.shape[0]):
                x1 = int(i-s/2)
                x2 = int(i+s/2)
                y1 = int(j-s/2)
                y2 = int(j+s/2)

                #bound validation
                if x2 >= arr.shape[1]:
                    x2 = arr.shape[1]-1

                if y2 >= arr.shape[0]:
                    y2 = arr.shape[0]-1


                #count = (x2 - x1) * (y2 - y1)
                x1-=1
                y1-=1
                if(x1 < 0):
                    x1 = 0

                if(y1 < 0):
                    y1 = 0

                count = (x2 - x1) * (y2 - y1)
                tot = intImg[y2, x2] - intImg[y2, x1] - intImg[y1, x2] + intImg[y1, x1]
                if (arr[j, i] * count) <= (tot * (100 - t)/100):
                    out[j, i] = 0
                else:
                    out[j,i] = 1
        toImg = copy.copy(out)
        toImg[toImg == 1] = 255
        self.binary_image = out
        self.image = Image.fromarray(toImg)


        if verbose:
            info["integral_image"] = intImg.tolist()
            img = self.image.convert("L")
            img.save("./static/cache/"+self.prefix+"binary.jpg")
            info["link"] = "./static/cache/"+self.prefix+"binary.jpg"
            return info

    def line_segmentation(self):
        arr = np.array(self.image)
        #horizontal projection
        projection = np.sum(arr==0,axis=1)
        info = {}
        info["projection"] = projection
        #get non zero index
        line = np.flatnonzero(projection)
        lines = []
        member = []
        number = 0

        for i in line:
            #check with threshold (noise)
            #if projection[i] < 4:
            #    if member and len(member) > 1:
            #        image = Image.fromarray(arr[member[0]:member[-1]]).convert("L")
            #        image.save("./static/cache/"+self.prefix+"line_segmentation_"+str(number)+".jpg")
            #        number+=1
            #        lines.append(member)
            #    member = []
            #    continue
            if not member:
                member.append(i)
                continue
            if member[-1] != i-1:
                if len(member) > 1:
                    image = Image.fromarray(arr[member[0]:member[-1]]).convert("L")
                    image.save("./static/cache/"+self.prefix+"line_segmentation_"+str(number)+".jpg")
                    number+=1
                    lines.append(member)
                member = [i]
            else:
                member.append(i)

            #last iteration
            if i == line[-1]:
                if member and len(member) > 2:
                    image = Image.fromarray(arr[member[0]:member[-1]]).convert("L")
                    image.save("./static/cache/"+self.prefix+"line_segmentation_"+str(number)+".jpg")
                    number+=1
                    lines.append(member)

        self.lines = lines
        info["number"] = number

        return info

    def word_segmentation(self):
        word = []
        arr = np.array(self.image)


        no_line = 0
        count_word = []
        for k in self.lines:
            arrW = arr[k[0]:k[-1]]
            projection = np.sum(arrW==0,axis=0)

            line = np.flatnonzero(projection)
            word_threshold = math.ceil(arrW.shape[0]*1/3)

            #cari gaps
            asli = projection[line[0]:line[-1]+1]
            asli = np.where(asli==0)[0]
            member = []
            lines = []
            for i in asli:
                if not member:
                    member.append(i)
                    continue
                if member[-1] != i-1:
                    #drop if len <= 2
                    #if len(member) > 0:
                    lines.append(member)

                    member = [i]
                else:
                    member.append(i)

                #last iteration
                if i == asli[-1]:
                    #if member and len(member) > 0:
                    lines.append(member)
            if lines:
                word_threshold_max = int(len(max(lines,key=len)))
            else:
                word_threshold_max  = 0

            if (len(line)/len(projection))*100 > 16:
                if word_threshold > word_threshold_max:
                    word_threshold = int(word_threshold_max*0.8)

            lines = []
            member = []
            number = 0
            for i in line:
                if not member:
                    member.append(i)
                    continue
                if member[-1] not in range(i-word_threshold,i):
                    #drop if len <= 2
                    if len(member) > 1:
                        lines.append(member)
                        image = Image.fromarray(arrW[:,member[0]:member[-1]]).convert("L")
                        image.save("./static/cache/"+self.prefix+"word_segmentation_"+str(no_line)+"_"+str(number)+".jpg")
                        number += 1

                    member = [i]
                else:
                    member.append(i)

                #last iteration
                if i == line[-1]:
                    if member and len(member) > 1:
                        lines.append(member)
                        image = Image.fromarray(arrW[:,member[0]:member[-1]]).convert("L")
                        image.save("./static/cache/"+self.prefix+"word_segmentation_"+str(no_line)+"_"+str(number)+".jpg")
                        number += 1


            count_word.append(number)
            no_line += 1
            word.append(lines)

        self.words = word
        info = {}
        info["count"] = count_word

        return info

    def character_segmentation(self):
        char_arr = []
        for ln in range(0,len(self.lines)):
            arrW = self.binary_image[self.lines[ln][0]:self.lines[ln][-1]]
            kata = []

            #word
            for wr in range(0,len(self.words[ln])):
                #char segmentation
                arr = arrW[:, self.words[ln][wr][0]:self.words[ln][wr][-1]]
                projection = np.sum(arr==0,axis=0)
                line = np.flatnonzero(projection)
                member = []
                lines = []
                #char
                for i in line:
                    if not member:
                        member.append(i)
                        continue
                    if member[-1] != i-1:
                        if len(member) > 0:
                            dat = arr[:, member[0]:member[-1]+1]
                            #char_rat = dat.shape[1]/dat.shape[0]

                            #check ratio
                            # if(char_rat > self.char_ratio):
                            #    seg = self.char_seg_touch(dat,member,arrW=arr)
                            #    c = []
                            #    chr = self.flat_char(seg,c)
                            #    for a in chr:
                            #        dat = arr[:, a[0]:a[-1]+1]
                            #        d = self.clean_resize_char(dat)
                            #        if d is not None:
                            #            lines.extend(d)
                            # else:
                            #     #cleaning and resize
                            #    d = self.clean_resize_char(dat)
                            #    if d is not None:
                            #        lines.extend(d)
                            d = self.clean_resize_char(dat)
                            if d is not None:
                                lines.extend(d)

                        member = []
                    else:
                        member.append(i)

                    #last iteration
                    if i == line[-1]:
                        if member and len(member) > 0:
                            dat = arr[:, member[0]:member[-1]+1]
                            #char_rat = dat.shape[1]/dat.shape[0]

                            #check ratio
                            # if(char_rat > self.char_ratio):
                            #    seg = self.char_seg_touch(dat,member,arrW=arr)
                            #    c = []
                            #    chr = self.flat_char(seg,c)
                            #    for a in chr:
                            #        dat = arr[:, a[0]:a[-1]+1]
                            #        d = self.clean_resize_char(dat)
                            #        if d is not None:
                            #            lines.extend(d)
                            # else:
                            #     #cleaning and resize
                            #    d = self.clean_resize_char(dat)
                            #    if d is not None:
                            #        lines.extend(d)
                            d = self.clean_resize_char(dat)
                            if d is not None:
                                lines.extend(d)

                if lines != []:
                    kata.append(lines)
                #print(kata[0])
                #break
            char_arr.append(kata)
            #break

        return char_arr

    def skew_corrected(self):
        delta = 0.5
        limit = 5
        angles = np.arange(-limit, limit+delta, delta)
        scores = []
        for angle in angles:
            hist, score = self.find_score(self.image, angle)

            scores.append(score)

        best_score = max(scores)
        best_angle = angles[scores.index(best_score)]

        # correct skew
        data = inter.rotate(self.image, best_angle, mode="nearest", reshape=False, order=0)
        img = Image.fromarray(data.astype("uint8"))
        self.image = img
        data[data == 255] = 1
        self.binary_image = data.astype("uint8")
        img.convert("RGB").save("./static/cache/"+self.prefix+"skew_corrected.jpg")

        return best_angle

    #skew score
    def find_score(self,arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score
    #char seg touch two way
    def char_seg_touch(self,dat,i,end=None,start=None,second=False,arrW=None):
        if end:
            #print(end)
            i = i[0:end]
        if start:
            i = i[start:]

        #get vertical projection
        projection = np.sum(dat==0,axis=0)

        #get the minima
        if len(projection) > 2:
            index_min = np.where(projection == projection[1:-1].min())[0]
        else:
            index_min = np.where(projection == projection.min())[0]


        #slicing should not in the first index
        index_min = index_min[index_min>0]

        #get the peak before minima
        before = np.array([projection[i-1] for i in index_min])
        #print(projection,index_min,before)
        point = index_min[np.where(before == before.max())[0][0]]

        #split right
        #right
        right = i[point:]
        #left
        left = i[0:point]
        #get len
        dat_left = arrW[:, left]
        dat_right = arrW[:, right]
        len_left = dat_left.shape[1]/dat_left.shape[0]
        len_right = dat_right.shape[1]/dat_right.shape[0]
        #print(len_left,len_right);
        if len_left > 0.5:
            left = self.char_seg_touch(dat_left,i,point,second=True,arrW=arrW)

        if len_right > 0.5:
            right = self.char_seg_touch(dat_right,i,start=point,second=True,arrW=arrW)

        if second:
            return left,right
        return [left]+[right]

    #flattening after segmenting touching character
    def flat_char(self,a,c):
        for b in a:
            if isinstance(b,tuple):
                self.flat_char(b,c)
            else:
                c.append(b)

        return c

    def clean_resize_char(self,dat):
        #clean
        pr = np.sum(dat==0,axis=1)
        l = np.flatnonzero(pr)
        d = None

        #resize
        if(len(l) > 1):
            image = Image.fromarray(dat[l[0]:l[-1]]).resize((self.char_size,self.char_size))
            d = np.array(image).astype("int").reshape(self.char_reshape).reshape(1,-1)

        return d
