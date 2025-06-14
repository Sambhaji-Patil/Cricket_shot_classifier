from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from utils import process_images

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        images = request.files.getlist('images')
        for img in images:
            filename = secure_filename(img.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(img_path)
            pred = process_images(img_path, app.config['UPLOAD_FOLDER'])
            results.append(pred)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)