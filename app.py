import os
from flask import Flask, request, render_template_string, redirect, url_for
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import docx
import zipfile
import shutil
import google.generativeai as genai
import json
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
import pytesseract
import io

# --- Gemini API Configuration ---
# The API key is read from an environment variable for security.
# Make sure to set `export GEMINI_API_KEY="YOUR_API_KEY"` in your terminal.
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Helper Functions for Text Extraction ---

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    try:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()

        # If text is minimal, it's likely a scanned PDF. Use OCR as a fallback.
        if len(text.strip()) < 100:
            print("--- Minimal text found. Attempting OCR fallback. ---")
            ocr_text = ""
            with fitz.open(file_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    # Render page to an image
                    pix = page.get_pixmap(dpi=300)
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    # Perform OCR on the image
                    ocr_text += pytesseract.image_to_string(img)
            return ocr_text
        else:
            return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error reading DOCX: {e}"

def extract_text_from_txt(file_path):
    """Extracts text from a TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading TXT: {e}"

# --- AI Interaction with Gemini ---
def analyze_document_with_ai(text):
    """
    Sends the document text to the Gemini API for analysis and returns the result.
    """
    if text.strip() == "Unsupported file type.":
        return {"error": "This file type is not supported for analysis."}
    if not text or not text.strip():
        return {"error": "The document appears to be empty or contains no readable text."}

    # This is the prompt that instructs the model on its task.
    prompt = f"""
You are an expert home inspector.

Your task is INFORMATION EXTRACTION.

IMPORTANT RULES:

1. Extract EVERY issue, defect, recommendation, hazard,
repair need, fungus condition, moisture issue, pest issue,
or condition likely to lead to future damage.


3. If the report mentions 10 separate issues, return 10 entries.

4. Treat each inspection item separately.

5. Include future-risk conditions even if no visible damage exists.

6. Include:
   - fungus
   - moisture
   - leaks
   - cracks
   - loose fixtures
   - pest activity
   - conditions likely to lead to future damage
   - recommendations for replacement or repair

7. Never combine multiple issues into one.

Return JSON only.

Example:

Input:
Item 8A: Garage door damaged by fungus.
Item 10A: Surface fungus on kitchen shelf.
Item 10B: Toilet loose.

Output:

{{
  "problems":[
    {{
      "location":"Garage",
      "item":"8A",
      "category":"Fungus",
      "description":"Garage side door and jambs damaged by fungus.",
      "severity":"High"
    }},
    {{
      "location":"Kitchen",
      "item":"10A",
      "category":"Fungus",
      "description":"Surface fungus found on kitchen shelf due to previous leaks.",
      "severity":"Medium"
    }},
    {{
      "location":"Hall Bathroom",
      "item":"10B",
      "category":"Plumbing",
      "description":"Toilet is loose or improperly mounted.",
      "severity":"Medium"
    }}
  ]
}}

Document:

{text}
"""
    try:
        print("--- Sending text to Gemini API for analysis ---")
        model = genai.GenerativeModel('gemini-3.1-flash-lite')
        generation_config = {
            "temperature": 0,
            "top_p": 0.1,
            "top_k": 1
        }

        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        try:
            # The model sometimes wraps the JSON in markdown. We need to extract the raw JSON string.
            response_text = response.text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_string = response_text[json_start:json_end]
                return json.loads(json_string)
            else:
                # If we can't find a JSON object, raise an error to be caught below.
                raise json.JSONDecodeError("Could not find JSON object in response", response_text, 0)
        except json.JSONDecodeError:
            print(f"Error: AI did not return valid JSON. Response:\n{response.text}")
            return {"error": "AI response was not in a valid JSON format."}
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {"error": f"An error occurred: {e}"}


# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Use getlist to handle multiple files
        uploaded_files = request.files.getlist("file")
        if not uploaded_files or uploaded_files[0].filename == '':
            return redirect(request.url)

        all_results = []
        for file in uploaded_files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # If the uploaded file is a ZIP archive
                if filename.lower().endswith('.zip'):
                    extract_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"extracted_{os.path.splitext(filename)[0]}")
                    os.makedirs(extract_folder, exist_ok=True)
                    
                    try:
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_folder)
                        
                        # Process each file in the extracted folder
                        for extracted_filename in os.listdir(extract_folder):
                            extracted_file_path = os.path.join(extract_folder, extracted_filename)
                            if os.path.isfile(extracted_file_path):
                                text = process_single_file(extracted_file_path, extracted_filename)
                                analysis_result = analyze_document_with_ai(text)
                                # Pass the raw dictionary to the template for structured display
                                all_results.append({'filename': f"{filename} -> {extracted_filename}", 'result': analysis_result})
                    finally:
                        # Clean up the extracted files and folder
                        shutil.rmtree(extract_folder)
                
                # If it's a regular, single file
                else:
                    text = process_single_file(file_path, filename)
                    analysis_result = analyze_document_with_ai(text)
                    # Pass the raw dictionary to the template for structured display
                    all_results.append({'filename': filename, 'result': analysis_result})


        # Display the results for all processed files
        return render_template_string(HTML_TEMPLATE, results=all_results)

    # Initial page load
    return render_template_string(HTML_TEMPLATE, results=None)

def process_single_file(file_path, filename):
    """Helper function to extract text from a single file based on its extension."""
    text = ""
    if filename.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif filename.lower().endswith('.docx'):
        text = extract_text_from_docx(file_path)
    elif filename.lower().endswith('.txt'):
        text = extract_text_from_txt(file_path)
    else:
        # This message will be shown for unsupported files inside a zip
        # or for unsupported single uploads.
        text = "Unsupported file type."
    return text

# --- HTML Template ---
# For simplicity, the HTML is included in the Python file.
# In a larger app, you would save this in a 'templates/index.html' file.
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Disclosure AI Uploader</title>
    <style>
        body { font-family: sans-serif; background-color: #f4f4f9; color: #333; margin: 40px; }
        .container { max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #0056b3; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        thead { background-color: #0056b3; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .severity-high { background-color: #f8d7da; color: #721c24; font-weight: bold; }
        .severity-medium { background-color: #fff3cd; color: #856404; font-weight: bold; }
        .severity-low { background-color: #d4edda; color: #155724; }
        .no-problems {
            padding: 15px;
            margin-top: 20px;
            background-color: #d4edda;
            color: #155724;
            border-left: 5px solid #28a745;
            font-weight: bold;
        }
        input[type=file], input[type=submit] { margin-top: 10px; padding: 8px; }
        .result { margin-top: 20px; padding: 15px; background: #e9ecef; border-left: 5px solid #0056b3; white-space: pre-wrap; }
        /* --- Loading Spinner Styles --- */
        #loader-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 9999;
            display: none; /* Hidden by default */
            justify-content: center;
            align-items: center;
            flex-direction: column;
            color: white;
        }
        .spinner {
            border: 8px solid #f3f3f3;
            border-top: 8px solid #0056b3;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div id="loader-overlay">
        <div class="spinner"></div>
        <p style="margin-top: 20px;">Analyzing documents, please wait...</p>
    </div>
    <div class="container">
        <h1>Upload Disclosure Document</h1>
        <p>Click the button below to select a folder containing your disclosure documents.</p>
        <p>You can select multiple files by holding down Shift or Ctrl/Cmd.</p>
        <form id="upload-form" method=post enctype=multipart/form-data>
            <input type="file" name="file" webkitdirectory multiple>
            <input type=submit value=Upload and Analyze>
        </form>
        {% if results %}
            {% for item in results %}
                <div class="result" style="white-space: normal;">
                    <h2>Analysis for: {{ item.filename }}</h2>
                    {% if item.result.get('problems') and item.result.get('problems')|length > 0 %}
                        <table>
                            <thead>
                                <tr>
                                    <th>Location</th>
                                    <th>Item</th>
                                    <th>Category</th>
                                    <th>Description</th>
                                    <th>Severity</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for problem in item.result.problems %}
                                <tr>
                                    <td>{{ problem.get('location', 'N/A') }}</td>
                                    <td>{{ problem.get('item', 'N/A') }}</td>
                                    <td>{{ problem.get('category', 'N/A') }}</td>
                                    <td>{{ problem.get('description', 'N/A') }}</td>
                                    <td class="severity-{{ problem.get('severity', '')|lower }}">{{ problem.get('severity', 'N/A') }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    {% elif item.result.get('error') %}
                        <p class="severity-high">Error: {{ item.result.get('error') }}</p>
                    {% else %}
                        <p class="no-problems">No problems were identified in this document.</p>
                    {% endif %}
                </div>
            {% endfor %}
        {% endif %}
    </div>
    <script>
        document.getElementById('upload-form').addEventListener('submit', function() {
            // Show the loader when the form is submitted
            const loader = document.getElementById('loader-overlay');
            loader.style.display = 'flex';
        });
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    # Runs the web server. Access it at http://127.0.0.1:5000
    app.run(host='0.0.0.0', port=5001, debug=True)
