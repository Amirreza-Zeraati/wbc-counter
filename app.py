import os
import cv2
from config import Config
from werkzeug.utils import secure_filename
from processing.detector import detect_and_count
from flask import Flask, render_template, request, jsonify, url_for

app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = cv2.imread(filepath)

        final_mask, count = detect_and_count(image)

        result_filename = f'processed_{filename}'
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        cv2.imwrite(result_path, final_mask)

        return jsonify({
            'success': True,
            'original': url_for('static', filename=f'uploads/{filename}'),
            'processed': url_for('static', filename=f'uploads/results/{result_filename}'),
            'count': int(count)
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
