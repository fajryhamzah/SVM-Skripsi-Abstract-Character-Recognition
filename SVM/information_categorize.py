class InformationCategorize:
    def __init__(self,txt):
        self.txt = txt
        self.judul_start = 1
        self.judul_end = 0
        self.nama = 0
        self.nim = 0
        self.keyword_end = len(txt)-1
        self.keyword_start = 0
        self.isi_start = 0
        self.isi_end = 0

        if self.judul_start >= len(txt):
            self.judul_start = 0
            self.keyword_end = 0
        else:
            self.process_information()


    def process_information(self):
        max_line = 5
        ln = []
        len_of_char = []
        i = self.judul_start+1
        for i in range(self.judul_start+1,len(self.txt)):
            split = self.txt[i]
            if "oleh" in split[0].lower() or "by" in split[0].lower():
                break

            #check the colon
            if len(split) == 2:
                if len(split[1]) == 1:
                    break

            ln.append(i)
            len_of_char.append(len("".join(split)))
            if i-self.judul_start > max_line:
                #most minimal
                i = ln[len_of_char.index(min(len_of_char))]
                break
        self.judul_end = i
        self.nama = self.judul_end+1
        self.nim = self.nama+1

        #keyword
        self.keyword_start = self.keyword_end
        if "kata kunci" not in " ".join(self.txt[self.keyword_end]).lower() and "keyword" not in " ".join(self.txt[self.keyword_end]).lower():
            if len("".join(self.txt[self.keyword_end])) < len("".join(self.txt[self.keyword_end-1])):
                self.keyword_start = self.keyword_end-1

        #isi
        self.isi_start = self.nim+1
        self.isi_end = self.keyword_start-1



    def get_all(self):
        return {"judul_start":self.judul_start,"judul_end":self.judul_end,"nama":self.nama,"nim":self.nim,"isi_start":self.isi_start,"isi_end":self.isi_end,"keyword_start":self.keyword_start,"keyword_end":self.keyword_end}
