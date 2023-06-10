from flask import Flask, render_template, request, jsonify, send_file
from model import split_text, analyze_text, create_new_doc_with_gpt_comments
import os
import json
import uuid
import threading
import queue

app = Flask(__name__)

analyzed_text_queue = queue.Queue()
analysis = []

def analyze_text_thread(input_text, labels):
    for chunk in analyze_text(input_text, labels):
        analyzed_text_queue.put(chunk)
    analyzed_text_queue.put([None, None])

@app.route('/')
def index():
    return render_template('index.html.j2')

@app.route('/analyze', methods=['POST'])
def analyze():
    global analysis
    analysis = []
    input_text = request.form['input_text']
    labels = json.loads(request.form['labels'])
    try:
        input_text = input_text.replace('\r', '')
        chunks = split_text(input_text)
        total_chunks = len(chunks)
        if not total_chunks:
            return jsonify(success=False)
    except RuntimeError:
        return jsonify(success=False)
    threading.Thread(target=analyze_text_thread, args=(input_text, labels)).start()
    return jsonify(success=True, total_chunks=total_chunks, chunks=chunks)

@app.route('/get_chunk', methods=['GET'])
def get_chunk():
    global analysis
    chunk, delimiter = analyzed_text_queue.get()
    if chunk is None:
        return jsonify(end=True)
    analysis.append(chunk)
    return jsonify(chunk=chunk, delimiter=delimiter)

@app.route('/download', methods=['GET'])
def download():
    global analysis
    filepath = os.path.join('.', 'analyzed_files', str(uuid.uuid4()) + '.docx')
    analyzed_doc = create_new_doc_with_gpt_comments(analysis)
    analyzed_doc.save(filepath)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
