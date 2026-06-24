import os
from flask import Flask, request, render_template_string, redirect, url_for, session, Response
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import docx
import zipfile
import shutil
import google.generativeai as genai
import json
from PIL import Image
import pytesseract
import csv
from datetime import datetime
import io

# --- Gemini API Configuration ---
# The API key is read from an environment variable for security.
# Make sure to set `export GEMINI_API_KEY="YOUR_API_KEY"` in your terminal.
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# A secret key is required to use sessions in Flask
app.secret_key = os.urandom(24)
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


2. If the report mentions 10 separate issues, return 10 entries.

3. Treat each inspection item separately.

4. Include future-risk conditions even if no visible damage exists.

5. Include:
   - fungus
   - moisture
   - leaks
   - cracks
   - loose fixtures
   - pest activity
   - conditions likely to lead to future damage
   - recommendations for replacement or repair

6. Never combine multiple issues into one.

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

        response = model.generate_content(prompt, generation_config=generation_config)
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

DOCUMENT_TYPES = {
    'tds': {'name': 'Transfer Disclosure Statement', 'required': True, 'description': 'Standard state-mandated disclosure form.'},
    'spq': {'name': 'Seller Property Questionnaire', 'required': True, 'description': 'Detailed questionnaire filled out by the seller.'},
    'nhd': {'name': 'Natural Hazard Disclosure Report', 'required': True, 'description': 'Report on natural hazards affecting the property.'},
    'prelim': {'name': 'Preliminary Report', 'required': True, 'description': 'Provides details on title, liens, and encumbrances.'},
    'inspection': {'name': 'Home Inspection Report', 'required': False, 'description': 'Professional home inspection findings.'},
    'non_foreign': {'name': "Seller's Affidavit of Non-Foreign Status", 'required': False, 'description': 'FIRPTA compliance form.'}
}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'uploaded_docs' not in session:
            session['uploaded_docs'] = {}

        for doc_key, file in request.files.items():
            if file and file.filename:
                filename = secure_filename(file.filename)
                # Ensure a unique folder for this session/upload batch
                session_folder = session.get('session_folder')
                if not session_folder:
                    session_folder = datetime.now().strftime("%Y%m%d%H%M%S%f")
                    session['session_folder'] = session_folder
                
                save_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_folder)
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, filename)
                file.save(file_path)

                session['uploaded_docs'][doc_key] = {
                    'filename': filename,
                    'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'path': file_path,
                    'source': 'Manual'
                }
        
        # This is where you would trigger analysis. For now, we just show the checklist.
        return redirect(url_for('analyze_all'))

    # Initial page load
    # For a new session, clear old data
    if request.method == 'GET' and not request.args:
        session.pop('uploaded_docs', None)
        session.pop('session_folder', None)

    uploaded_docs = session.get('uploaded_docs', {})
    
    # Calculate progress statistics
    total_docs = len(DOCUMENT_TYPES)
    required_docs = {k:v for k,v in DOCUMENT_TYPES.items() if v['required']}
    
    loaded_count = len(uploaded_docs)
    required_loaded_count = sum(1 for k in required_docs if k in uploaded_docs)
    
    progress_percent = int((loaded_count / total_docs) * 100) if total_docs > 0 else 0
    all_required_loaded = required_loaded_count == len(required_docs)

    stats = {'total_docs': total_docs, 'loaded_count': loaded_count, 'progress_percent': progress_percent, 'all_required_loaded': all_required_loaded}

    return render_template_string(HTML_TEMPLATE, doc_types=DOCUMENT_TYPES, uploaded_docs=uploaded_docs, stats=stats)

@app.route('/analyze')
def analyze_all():
    # This is a placeholder for the analysis logic.
    # In a real app, this would iterate through session['uploaded_docs'],
    # run analysis on each, and display a results page.
    # For now, let's just show the uploaded files.
    uploaded_docs = session.get('uploaded_docs', {})
    if not uploaded_docs:
        return redirect(url_for('upload_file'))

    all_results = []
    for doc_key, doc_info in uploaded_docs.items():
        text = process_single_file(doc_info['path'], doc_info['filename'])
        analysis_result = analyze_document_with_ai(text)
        all_results.append({'filename': f"{DOCUMENT_TYPES[doc_key]['name']} ({doc_info['filename']})", 'result': analysis_result})
    
    # Save results to session so they can be downloaded
    session['analysis_results'] = all_results
    
    stats = _get_progress_stats()
    uploaded_docs = session.get('uploaded_docs', {})

    # Redirect to the main page to show results
    return render_template_string(HTML_TEMPLATE, doc_types=DOCUMENT_TYPES, uploaded_docs=uploaded_docs, results=all_results, stats=stats)

@app.route('/bulk_analyze', methods=['POST'])
def bulk_analyze():
    """Handles the simple folder upload and immediate analysis."""
    uploaded_files = request.files.getlist("bulk_files")
    if not uploaded_files or uploaded_files[0].filename == '':
        return redirect(url_for('upload_file'))

    all_results = []
    # Create a unique folder for this bulk upload to avoid filename collisions
    session_folder = datetime.now().strftime("%Y%m%d%H%M%S%f")
    save_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_folder)
    os.makedirs(save_dir, exist_ok=True)

    for file in uploaded_files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(save_dir, filename)
            file.save(file_path)

            text = process_single_file(file_path, filename)
            analysis_result = analyze_document_with_ai(text)
            all_results.append({'filename': filename, 'result': analysis_result})

    # Save results to session for downloading
    session['analysis_results'] = all_results
    
    # Render the main template, showing the results but with an empty checklist
    stats = _get_progress_stats() # This will show 0% progress for the checklist
    return render_template_string(HTML_TEMPLATE, doc_types=DOCUMENT_TYPES, uploaded_docs={}, results=all_results, stats=stats)

@app.route('/download_csv')
def download_csv():
    """Generates and serves a CSV file of the analysis results."""
    results = session.get('analysis_results', [])
    if not results:
        return redirect(url_for('upload_file'))

    # Use an in-memory string buffer
    output = io.StringIO()
    writer = csv.writer(output)

    # Write the header
    header = ['Source File', 'Location', 'Item', 'Category', 'Description', 'Severity']
    writer.writerow(header)

    # Write the data rows
    for item in results:
        filename = item.get('filename', 'N/A')
        if item.get('result', {}).get('problems'):
            for problem in item['result']['problems']:
                row = [
                    filename,
                    problem.get('location', 'N/A'),
                    problem.get('item', 'N/A'),
                    problem.get('category', 'N/A'),
                    problem.get('description', 'N/A'),
                    problem.get('severity', 'N/A')
                ]
                writer.writerow(row)

    output.seek(0)
    return Response(output, mimetype="text/csv", headers={"Content-Disposition":"attachment;filename=disclosure_analysis.csv"})

def _get_progress_stats():
    """Helper function to calculate and return upload progress stats."""
    uploaded_docs = session.get('uploaded_docs', {})
    total_docs = len(DOCUMENT_TYPES)
    required_docs = {k:v for k,v in DOCUMENT_TYPES.items() if v['required']}
    loaded_count = len(uploaded_docs)
    required_loaded_count = sum(1 for k in required_docs if k in uploaded_docs)
    progress_percent = int((loaded_count / total_docs) * 100) if total_docs > 0 else 0
    all_required_loaded = required_loaded_count == len(required_docs)
    return {'total_docs': total_docs, 'loaded_count': loaded_count, 'progress_percent': progress_percent, 'all_required_loaded': all_required_loaded}

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
        .progress-summary { text-align: center; margin-bottom: 30px; }
        .progress-bar { background-color: #e9ecef; border-radius: .25rem; height: 1rem; }
        .progress-bar-inner { background-color: #007bff; height: 1rem; border-radius: .25rem; }
        .section { border: 1px solid #dee2e6; border-radius: 5px; padding: 20px; margin-top: 30px; }
        .section h2 { margin-top: 0; }
        .section p { color: #6c757d; }
        .section input[type=text] { width: 100%; padding: 8px; margin-top: 10px; }
        .doc-table { width: 100%; margin-top: 20px; border-collapse: collapse; }
        .doc-table th, .doc-table td { padding: 12px; border-bottom: 1px solid #dee2e6; text-align: left; }
        .doc-table th { font-weight: bold; color: #495057; }
        .doc-list { list-style: none; padding: 0; }
        .doc-item { background: #f8f9fa; border: 1px solid #dee2e6; padding: 15px; margin-bottom: 10px; border-radius: 5px; display: flex; align-items: center; justify-content: space-between; }
        .doc-info h3 { margin: 0; font-size: 1.1em; }
        .doc-info p { margin: 5px 0 0; color: #6c757d; font-size: 0.9em; }
        .doc-status { text-align: right; }
        .status-uploaded { color: #28a745; font-weight: bold; }
        .status-missing { color: #dc3545; font-weight: bold; }
        .status-manual { color: #17a2b8; }
        .status-auto { color: #28a745; }
        .doc-actions .btn { padding: 5px 10px; font-size: 0.8em; cursor: pointer; }
        .btn-preview { background-color: #007bff; color: white; border: none; border-radius: 3px; }
        .btn-replace { background-color: #ffc107; color: black; border: none; border-radius: 3px; }
        h1, h2 { color: #0056b3; }
        .summary-box {
            display: flex;
            justify-content: space-around;
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .summary-item { text-align: center; }
        .summary-item h3 { margin: 0; font-size: 24px; }
        .summary-item p { margin: 0; color: #6c757d; }
        .summary-high { color: #721c24 !important; }
        .summary-medium { color: #856404 !important; }
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
        <h1>Upload Documents</h1>
        <p>Load property disclosure documents manually or use the disclosure link to auto-fetch available files.</p>

        <div class="progress-summary">
            <div class="progress-bar">
                <div class="progress-bar-inner" style="width: {{ stats.progress_percent }}%;"></div>
            </div>
            <p style="margin-top: 10px;"><b>{{ stats.progress_percent }}%</b></p>
            <p><b>{{ stats.loaded_count }} of {{ stats.total_docs }} Documents Loaded</b></p>
            <p>{{ stats.total_docs - stats.loaded_count }} documents remaining to analyze</p>
        </div>

        <div class="section">
            <h2>1. Auto Load from Disclosure Link</h2>
            <p>Paste the disclosure link below and we'll automatically fetch available documents.</p>
            <input type="text" placeholder="Paste disclosure link here...">
            <!-- In a real app, this would have JS to trigger a fetch -->
        </div>

        <div class="section">
            <h2>2. Manual Checklist Upload</h2>
            <p>Review, preview, or upload documents before running the analysis.</p>
            <form id="upload-form" method=post enctype=multipart/form-data>
                <table class="doc-table">
                    <thead>
                        <tr><th colspan="4">Required Documents</th></tr>
                        <tr>
                            <th>Document Type</th>
                            <th>Uploaded Document Name</th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for key, doc in doc_types.items() if doc.required %}
                        <tr>
                            <td><b>{{ doc.name }}*</b><br><small>{{ doc.description }}</small></td>
                            <td>{{ uploaded_docs[key].filename if uploaded_docs.get(key) else '—' }}</td>
                            <td>
                                {% if uploaded_docs.get(key) %}
                                    <span class="status-manual">Loaded (Manual)</span>
                                {% else %}
                                    <span class="status-missing">Missing</span>
                                {% endif %}
                            </td>
                            <td><input type="file" name="{{ key }}"></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                    <thead>
                        <tr><th colspan="4" style="padding-top: 30px;">Optional Documents</th></tr>
                    </thead>
                    <tbody>
                        {% for key, doc in doc_types.items() if not doc.required %}
                        <tr>
                            <td><b>{{ doc.name }}</b><br><small>{{ doc.description }}</small></td>
                            <td>{{ uploaded_docs[key].filename if uploaded_docs.get(key) else '—' }}</td>
                            <td>
                                {% if uploaded_docs.get(key) %}
                                    <span class="status-manual">Loaded (Manual)</span>
                                {% else %}
                                    <span>Not uploaded</span>
                                {% endif %}
                            </td>
                            <td><input type="file" name="{{ key }}"></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                <input type=submit value="Upload Selected Files">
            </form>
        </div>

        <div class="section">
            <h2>3. Bulk Folder Upload & Analyze</h2>
            <p>Select a folder, and we will immediately analyze all documents inside it.</p>
            <form id="bulk-upload-form" action="/bulk_analyze" method="post" enctype="multipart/form-data">
                <input type="file" name="bulk_files" webkitdirectory multiple>
                <input type="submit" value="Upload Folder and Analyze">
            </form>
        </div>

        {% if results %}
            <hr style="margin-top: 30px;">
            <h2>Analysis Results</h2>
            {% for item in results %}
                <div class="result" style="white-space: normal;">
                    <h2>Analysis for: {{ item.filename }}</h2>
                    {% if item.result.get('problems') and item.result.get('problems')|length > 0 %}
                        <table>
                            <thead>
                                <tr>
                                    <th>Location</th>
                                    <th>Item</th>
                                    <th>Description</th>
                                    <th>Severity</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for problem in item.result.problems %}
                                <tr>
                                    <td>{{ problem.get('location', 'N/A') }}</td>
                                    <td>{{ problem.get('item', 'N/A') }}</td>
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
            <a href="/download_csv" class="download-button">Download All Results as CSV</a>
        {% endif %}
    </div>
    <style>
        .download-button {
            display: inline-block;
            margin-top: 20px;
            padding: 10px 15px;
            background-color: #28a745;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
        }
    </style>
    <script>
        document.getElementById('upload-form').addEventListener('submit', function() {
            // Show the loader when the form is submitted
            const loader = document.getElementById('loader-overlay');
            loader.style.display = 'flex';
        });

        const bulkForm = document.getElementById('bulk-upload-form');
        if (bulkForm) {
            bulkForm.addEventListener('submit', function() {
                document.getElementById('loader-overlay').style.display = 'flex';
            });
        }

        const analyzeButton = document.getElementById('analyze-button');
        if (analyzeButton) {
            analyzeButton.addEventListener('click', function() {
                document.getElementById('loader-overlay').style.display = 'flex';
            });
        }
    </script>
</body>

</html>
"""

if __name__ == '__main__':
    # Runs the web server. Access it at http://127.0.0.1:5000
    app.run(host='0.0.0.0', port=5001, debug=True)
