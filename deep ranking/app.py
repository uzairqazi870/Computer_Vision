from flask import Flask, request, flash, Response, redirect, url_for, render_template
import numpy as np
from flask import jsonify
import os
import json
import pickle
from werkzeug.utils import secure_filename
from zipfile import ZipFile
import traceback
import json
from triplet_sampler import generate_triplets
from train_net import main
from predict import pred_main


UPLOAD_FOLDER = './upload_folder'
TRIPLET_PATH = "triplet.csv"
ALLOWED_EXTENSIONS = {'zip'}
MODEL_PATH = 'model/deeprank17.pt'
EMBEDDING_PATH = 'embedding.txt'
QUERY_IMG_PATH = './dataset/100s/charcoal_11_100s_negative.jpg'
epochs = 1
optimizer = 'adam' # 'sgd', 'rms'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize():
	if not os.path.exists(UPLOAD_FOLDER):
		os.makedirs(UPLOAD_FOLDER)
		print("Created upload folder")

	if not os.path.exists("training_data"):
		os.makedirs("training_data")
		print("Created training folder")

	if not os.path.exists("model"):
		os.makedirs("model")
		print("Created model folder")
		

initialize()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        val = request.form.getlist('mode')[0]

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
            	return redirect(url_for('inference', filename=filename))
            elif val == "training":
            	return redirect(url_for('training', filename=filename))

    return render_template("index.html")


@app.route("/dp_train/<filename>", methods=['POST','GET'])
def training(filename):

	try:	
		dataZipPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		ZipFile(dataZipPath).extractall("training_data")

		training_path = "./training_data/"
		dataFolder = os.path.join(training_path, filename[:-4])

		if not os.path.exists(TRIPLET_PATH):
			print("==> generating triplets")
			generate_triplets(dataset_path=dataFolder, TRIPLET_PATH=TRIPLET_PATH, num_neg_images=1, num_pos_images=1)

		else:
			print("==> found existing triplets file")

		result = main(EPOCHS=epochs, OPTIM_NAME=optimizer, TRAIN_PATH= 'model/checkpoint/deeprank', TRIPLET_PATH=TRIPLET_PATH)


		if result == "success":
			final_result = {"status":"Success", "Result":"Training completed"}
		else:
			final_result = {"status":"Failed", "Result":"Training incomplete"}

		return jsonify(final_result)


	except Exception as e:
		print("Exception :",e)
		print(traceback.print_exc())
		return jsonify({'exception':500})


@app.route("/dp_infer<filename>", methods=['POST','GET'])
def inference(filename):

	try:
		dataZipPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		ZipFile(dataZipPath).extractall("training_data")

		training_path = "./training_data/"
		dataFolder = os.path.join(training_path, filename[:-4])

		if not os.path.exists(TRIPLET_PATH):
			print("==> generating triplets")
			generate_triplets(dataset_path=dataFolder, TRIPLET_PATH=TRIPLET_PATH, num_neg_images=1, num_pos_images=1)
		else:
			print("==> found existing triplets file")

		train_result = main(EPOCHS=epochs, OPTIM_NAME=optimizer, TRAIN_PATH= 'model/checkpoint/deeprank', TRIPLET_PATH=TRIPLET_PATH)

		if train_result == "success":
			result = pred_main(MODEL_PATH, EMBEDDING_PATH, TRIPLET_PATH, image_path = QUERY_IMG_PATH)
		else:
			result = "training failed"
		

		if result == "success":
			final_result = {"status":"Success", "Result":"inference saved in same directory !"}
		elif result == "training failed":
			final_result = {"status":"Failed", "Result":"training failed!"}
		else:
			final_result = {"status":"Failed", "Result":"inference incomplete"}

		return jsonify(final_result)


	except Exception as e:
		print("Exception :",e)
		print(traceback.print_exc())
		return jsonify({'exception':500})

if __name__ == "__main__":
	app.run(debug=True)