from flask import Flask, request, flash, Response, redirect, url_for, render_template
import numpy as np
from flask import jsonify
import os
import json
import pickle
from werkzeug.utils import secure_filename

from imagededup.methods import CNN
from zipfile import ZipFile
import traceback
import json

UPLOAD_FOLDER = './upload_folder'
ALLOWED_EXTENSIONS = {'zip'}

def initialize():
	print("loading the model ...")
	global method_object
	method_object = CNN()

	if not os.path.exists(UPLOAD_FOLDER):
		os.makedirs(UPLOAD_FOLDER)
		print("Created upload folder")

	if not os.path.exists("inference_data"):
		os.makedirs("inference_data")
		print("Created inference folder")

	if not os.path.exists("training_data"):
		os.makedirs("training_data")
		print("Created training folder")


initialize()


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        val = request.form.getlist('mode')[0]

        thresh = request.form.getlist('thresh')[0]
        print(thresh)

        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            if val == "inference":
            	return redirect(url_for('inference', filename=filename, thresh=thresh))
            elif val == "training":
            	return redirect(url_for('training', filename=filename, thresh=thresh))

    return render_template("index.html")




@app.route("/dd_infer/<filename>/<thresh>", methods=['POST','GET'])
def inference(filename, thresh):

	try:

		dataZipPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
	
		ZipFile(dataZipPath).extractall("inference_data")

		inference_path = "./inference_data/"
		dataFolder = os.path.join(inference_path, filename[:-4])
		#print(dataFolder)
		duplicates = method_object.find_duplicates_to_remove(image_dir=dataFolder, min_similarity_threshold=float(thresh), outfile="result.json")
		
		# for dup in duplicates:
		# 	img_path = os.path.join(dataFolder, dup)
		# 	print(img_path)

		return jsonify(duplicates)


	except Exception as e:
		print("Exception :",e)
		print(traceback.print_exc())
		return jsonify({'exception':500})

@app.route("/dd_train/<filename>", methods=['POST','GET'])
def training(filename):

	try:

		dataZipPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		ZipFile(dataZipPath).extractall("training_data")

		training_path = "./training_data/"
		dataFolder = os.path.join(training_path, filename[:-4])

		encodings = None

		encodings = method_object.encode_images(image_dir=dataFolder)

		# for enc in encodings.keys():
  # 			encodings[enc] = encodings[enc].tolist()

		# Serialize data into file:
		# with open("encodings.json","w") as file:
		#   json.dump(encodings, file)

		with open("encodings.pkl","wb") as file:
			pickle.dump(encodings, file, pickle.HIGHEST_PROTOCOL)

		if encodings:
			final_result = {"status":"Success", "Result":"Training completed"}
		else:
			final_result = {"status":"Failed", "Result":"Training incomplete"}

		return jsonify(final_result)


	except Exception as e:
		print("Exception :",e)
		print(traceback.print_exc())
		return jsonify({'exception':500})


if __name__ == "__main__":
	app.run(debug=False)