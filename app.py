import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import fitz  # PyMuPDF

app = Flask(__name__)  

# Initialize the Pegasus tokenizer and model
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

# Define the directory where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed file extensions (e.g., PDF, TXT)
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    uploaded_file = request.files['file']
    min_len = int(request.form['min_length'])  # Retrieve max length from the form
    max_len = int(request.form['max_length'])  # Retrieve min length from the form
    if 'file' not in request.files or uploaded_file.filename == '':
        return render_template('index.html', error='No selected file')

    if uploaded_file and allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)

        pdf_document = fitz.open(file_path)
        pdf_text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            pdf_text += page.get_text()

        tokens = tokenizer(pdf_text, truncation=True, padding="longest", return_tensors="pt")

        summary_ids = model.generate(tokens.input_ids, max_length=max_len, min_length=min_len, length_penalty=2.0, num_beams=4, early_stopping=True)

        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Remove <pad> and <s> tokens from the summary
        summary_text = summary_text.replace('<pad>', '').replace('<s>', '')

        return render_template('index.html', input_text=pdf_text, summary_text=summary_text)

    else:
        return render_template('index.html', error='Invalid file format. Please upload a PDF or TXT file.')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)


