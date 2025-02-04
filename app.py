from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Charger le modèle GPT-2 et le tokenizer
model_name = 'gpt2'  # Tu peux aussi utiliser d'autres variantes comme 'gpt2-medium' si tu veux plus de puissance
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Fonction pour générer une légende pour une image
def generate_caption(image_name):
    # Améliorer le prompt pour être plus descriptif et pertinent
    prompt = f"Describe a beautiful piece of artwork named '{image_name}' in detail."
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Générer du texte
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Limiter à une légende simple, par exemple : "A beautiful painting of nature."
    return caption


# Initialisation de l'application Flask
app = Flask(__name__)

# Configuration pour les uploads
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['AUDIO_FOLDER'] = 'static/audio'
app.config['IMAGE_FOLDER'] = 'static/image'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['ALLOWED_AUDIO_EXTENSIONS'] = {'mp3', 'wav', 'ogg'}

# Vérifier si un fichier est autorisé
def allowed_file(filename, file_type):
    if file_type == "image":
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_IMAGE_EXTENSIONS']
    elif file_type == "audio":
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_AUDIO_EXTENSIONS']
    return False

# Page d'accueil
@app.route("/")
def home():
    return render_template("index.html")

# Galerie d'art (Images)
@app.route("/gallery")
def gallery():
    image_files = [f for f in os.listdir(app.config['IMAGE_FOLDER']) if allowed_file(f, "image")]
    captions = {}

    # Générer une légende pour chaque image
    for image in image_files:
        caption = generate_caption(image)
        captions[image] = caption

    return render_template("gallery.html", images=image_files, captions=captions)


# Galerie audio
@app.route("/gallery_audio")
def gallery_audio():
    audio_files = [f for f in os.listdir(app.config['AUDIO_FOLDER']) if allowed_file(f, "audio")]
    return render_template("gallery_audio.html", audio_files=audio_files)

# Upload d'images
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename, "image"):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for("apply_filter", filename=filename))
    return render_template("upload.html")

# Appliquer des filtres à une image
@app.route("/apply_filter/<filename>")
def apply_filter(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(filepath)

    # Appliquer un filtre : ici deux filtres sont appliqués
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(img, (15, 15), 0)

    # Sauvegarder les images filtrées
    gray_path = os.path.join(app.config['IMAGE_FOLDER'], "gray_" + filename)
    blurred_path = os.path.join(app.config['IMAGE_FOLDER'], "blurred_" + filename)

    cv2.imwrite(gray_path, gray_img)
    cv2.imwrite(blurred_path, blurred_img)

    # Passer les images filtrées dans le template
    return render_template(
        "filtered.html",
        original=filename,
        gray="image/gray_" + filename,
        blurred="image/blurred_" + filename
    )


# Upload d'audio
@app.route("/upload_audio", methods=["GET", "POST"])
def upload_audio():
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename, "audio"):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['AUDIO_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for("play_audio", filename=filename))
    return render_template("upload_audio.html")

# Lecture d'audio
@app.route("/play_audio/<filename>")
def play_audio(filename):
    return render_template("play_audio.html", audio_file=filename)

# Visualisation des données
@app.route("/data_viz")
def data_viz():
    data = pd.DataFrame({"Year": [2010, 2011, 2012, 2013, 2014], "Sales": [100, 200, 300, 400, 500]})
    plt.figure()
    plt.plot(data["Year"], data["Sales"], marker="o", linestyle="-", color="blue")
    plt.title("Sales Over Time")
    plt.xlabel("Year")
    plt.ylabel("Sales (USD)")
    plt.grid(True)
    chart_path = os.path.join("static", "images", "sales_chart.png")
    plt.savefig(chart_path)
    plt.close()
    return render_template("data_viz.html", chart="images/sales_chart.png")

# Lancer l'application Flask
if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
    os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)
    app.run(debug=True)



