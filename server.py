from flask import Flask, request, redirect, flash, url_for, send_file
from werkzeug.utils import secure_filename
from flask import render_template
from SVM.main import Main
import os
import sys
import glob
from random import randint

app = Flask(__name__)
app.secret_key = 'some_secret'
allowed_file = ["png","jpeg","jpg"]
upload_folder = "./static/image/"
app.config['UPLOAD_FOLDER_CHARACTER'] = upload_folder+"character_test"
app.config['UPLOAD_FOLDER_TRAIN'] = upload_folder+"train_set"
app.config['UPLOAD_FOLDER'] = upload_folder+"abstrak"
app.config['UPLOAD_FOLDER_TEST'] = upload_folder+"character_test"
app.config['MODEL_FOLDER'] = "model"

def allowed(filename):
    allow = set(allowed_file)
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allow

def redirect_url(default='/'):
    return request.args.get('next') or \
           request.referrer or \
           url_for(default)

def set_model():
    MAIN.set_model(app.config['MODEL_ACTIVE'])

@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/")
def hello():
    Main()
    #print(MAIN.model_img_size)
    info = {}
    info["list_img"] = MAIN.list_img_uploaded()
    return render_template('home.html', title='Home', info=info)

@app.route("/download/<name>")
def dl(name):
    name = name.split(".")[0]+"_.csv"
    return send_file("./static/csv/"+name, as_attachment=True)

@app.route("/delete/<name>")
def delete(name):
    name = name.split(".")[0]
    #file inti
    fileList = glob.glob("./static/image/abstrak/"+name+".*")
    if os.path.exists(fileList[0]):
        os.remove(fileList[0])

    #file csv
    fileList = "./static/csv/"+name+"_.csv"
    if os.path.exists(fileList):
        os.remove(fileList)

    #file answer
    fileList = "./static/answer/"+name+".txt"
    if os.path.exists(fileList):
        os.remove(fileList)

    #file cache
    fileList = glob.glob("./static/cache/*"+name+"_*")
    for filePath in fileList:
        if os.path.exists(filePath):
            os.remove(filePath)

    return redirect("/")

@app.route("/training")
def train():
    info = {}
    info["karakter"] = MAIN.model.model.classes_
    info["kernel"] = MAIN.model.kernel
    #print(MAIN.model.model)
    return render_template('training.html', title='Training', info=info)

@app.route("/train",methods=["post"])
def traintest():
    info = {}
    type = request.form.get("type")
    info = {}

    if type == "single":
        label = request.form.get("label")
        if 'file' not in request.files:
            flash('No file part')
            return redirect("/training")

        file = request.files['file']

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect("/training")

        if file and allowed(file.filename):
            filename = secure_filename(file.filename)
            clean = os.path.join(app.config['UPLOAD_FOLDER_TRAIN'], filename)
            file.save(clean)
        else:
            flash('Format file tidak didukung(png,jpeg,jpg)')
            return redirect("/training")

        if request.form.get("label1"):
            label1 = request.form.get("label1")
            file1 = request.files['file1']
            if label == label1:
                flash('Kelas/Label tidak boleh sama')
                return redirect("/training")

            if file1.filename == '':
                flash('No selected file on second file')
                return redirect("/training")

            if file1 and allowed(file1.filename):
                filename1 = secure_filename(file1.filename)
                clean1 = os.path.join(app.config['UPLOAD_FOLDER_TRAIN'], filename1)
                file1.save(clean1)
            else:
                flash('Format file kedua tidak didukung(png,jpeg,jpg)')
                return redirect("/training")

        if request.form.get("label1"):
            info = MAIN.single_train(clean,label,clean1,label1)
        else:
            info = MAIN.single_train(clean,label)
    else:
        if 'file_csv' not in request.files:
            flash('No file part')
            return redirect("/training")

        c = int(request.form.get("c"))
        kernel = request.form.get("kernel")
        gamma = request.form.get("gamma")
        file = request.files['file_csv']

        if file.filename == '':
            flash('No selected file on dataset field')
            return redirect("/training")

        if file.filename.split(".")[1] == "csv":
            filename = secure_filename(file.filename)

            if not os.path.exists(app.config['MODEL_FOLDER']+"/"+filename.split(".")[0]):
                os.mkdir(app.config['MODEL_FOLDER']+"/"+filename.split(".")[0])

            clean = os.path.join(app.config['MODEL_FOLDER']+"/"+filename.split(".")[0], filename)

            file.save(clean)
        else:
            flash('Format file tidak didukung(csv)')
            return redirect("/training")

        if int(gamma) < 0:
            gamma = "scale"
        else:
            gamma = int(gamma)

        info = MAIN.bulk_train(kernel,c,gamma,filename.split(".")[0])
        app.config["MODEL_ACTIVE"] = filename.split(".")[0]
        MAIN.set_model(app.config["MODEL_ACTIVE"])
        msg = "Model baru berhasil dilatih ("+info["name"]+").<br/>Kernel: "+str(info["kernel"])+" <br/>Jumlah data: "+str(info["data"])+"<br/>Jumlah kelas: "+str(info["class"])
        flash(msg)
        return redirect("/training")

    return render_template('trainingcoba.html', title='Training', info=info)

@app.route("/train/save",methods=["get"])
def trainsave():
    MAIN.classifier_save()
    flash('Hasil pelatihan telah disimpan')
    return redirect("/training")

@app.route("/train/cancel",methods=["get"])
def traincancel():
    MAIN.classifier_rollback()
    flash('Hasil pelatihan tidak disimpan')
    return redirect("/training")

@app.route('/change_model/<name>')
def change_model(name):
    if name in app.config['MODEL_LIST']:
        app.config['MODEL_ACTIVE'] = name
        MAIN.set_model(app.config['MODEL_ACTIVE'])
    return redirect(redirect_url())

@app.route("/uji")
def uji():
    info = {}
    info["list"] = MAIN.accuracy_list()

    if request.args.get("test"):
        clean = request.args.get("test")
        acc = MAIN.check_accuracy(clean)
        flash("Accuracy(sensitive) : "+str(acc["sensitive"])+"<br/>Accuracy(insensitive) : "+str(acc["insensitive"]))
    return render_template("uji.html", title="Uji Klasifier",info=info)

@app.route("/uji",methods=["post"])
def uji_process():
    info = {}

    if request.form.get("type"):
        if 'file_bulk' not in request.files:
            flash('No file part')
            return redirect("/uji")
        file = request.files['file_bulk']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect("/uji")

        if file.filename.split(".")[1] == "csv":
            filename = secure_filename(file.filename)
            clean = os.path.join(app.config['UPLOAD_FOLDER_TEST'], filename)
            file.save(clean)

        acc = MAIN.check_accuracy(filename)
        flash("Accuracy(sensitive) : "+str(acc["sensitive"])+"<br/>Accuracy(insensitive) : "+str(acc["insensitive"]))
        return redirect("/uji")



    if 'file' not in request.files:
        flash('No file part')
        return redirect("/uji")
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect("/uji")
    if file and allowed(file.filename):
        filename = secure_filename(file.filename)
        clean = os.path.join(app.config['UPLOAD_FOLDER_CHARACTER'], filename)
        file.save(clean)

    else:
        flash('Format file tidak didukung(png,jpeg,jpg)')
        return redirect("/uji")

    info = MAIN.classifier_test(clean)
    info['clean'] = clean
    #return user
    return render_template('uji_hasil.html', title='Test', info=info)

@app.route("/test",methods=["post","get"])
def test():
    info = {}
    skip = None
    compare = None
    if request.method == "GET":
        clean = request.args.get("img")
        skip = MAIN.text_list()

        if request.args.get("banding"):
            compare = request.args.get("banding")

        if request.args.get("retest"):
            skip = None
            clean = os.path.join(app.config['UPLOAD_FOLDER'], clean)
    else:
        if 'file' not in request.files:
            flash('No file part')
            return redirect("/")

        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect("/")
        if file and allowed(file.filename):
            filename = secure_filename(file.filename)
            clean = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(clean)
        else:
            flash('Format file tidak didukung(png,jpeg,jpg)')
            return redirect("/")

        if "file_compare" in request.files:
            file = request.files['file_compare']
            filename = secure_filename(file.filename)
            file.save("./static/answer/"+filename)
            compare = filename

    info = MAIN.test(clean,skip,compare)
    #return ''

    info['clean'] = clean
    #return user
    return render_template('test.html', title='Test', info=info)

MAIN = Main()
app.config['MODEL_LIST'] = MAIN.model_list()
app.config['MODEL_ACTIVE'] = app.config['MODEL_LIST'][0]
set_model()


if __name__ == "__main__":
    app.run(debug=True)
