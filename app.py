from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
from pathlib import Path
from handle_query import rag_query
from index_papers import index_papers
import config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'projects'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_projects():
    """List all project directories"""
    projects = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for d in os.listdir(app.config['UPLOAD_FOLDER']):
            project_path = os.path.join(app.config['UPLOAD_FOLDER'], d)
            if os.path.isdir(project_path):
                papers_path = os.path.join(project_path, 'papers')
                index_path = os.path.join(project_path, 'index.pkl')
                paper_count = len(os.listdir(papers_path)) if os.path.exists(papers_path) else 0
                indexed = os.path.exists(index_path)
                projects.append({
                    'name': d,
                    'paper_count': paper_count,
                    'indexed': indexed
                })
    return projects

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/projects', methods=['GET'])
def list_projects():
    return jsonify(get_projects())

@app.route('/api/projects', methods=['POST'])
def create_project():
    data = request.json
    project_name = secure_filename(data.get('name', ''))
    
    if not project_name:
        return jsonify({'error': 'Project name required'}), 400
    
    project_path = os.path.join(app.config['UPLOAD_FOLDER'], project_name)
    papers_path = os.path.join(project_path, 'papers')
    
    if os.path.exists(project_path):
        return jsonify({'error': 'Project already exists'}), 400
    
    os.makedirs(papers_path, exist_ok=True)
    return jsonify({'success': True, 'name': project_name})

@app.route('/api/projects/<project_name>/papers', methods=['POST'])
def upload_papers(project_name):
    project_name = secure_filename(project_name)
    papers_path = os.path.join(app.config['UPLOAD_FOLDER'], project_name, 'papers')
    
    if not os.path.exists(papers_path):
        return jsonify({'error': 'Project not found'}), 404
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    uploaded = []
    
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            if filename.lower().endswith('.pdf'):
                filepath = os.path.join(papers_path, filename)
                file.save(filepath)
                uploaded.append(filename)
    
    return jsonify({'success': True, 'uploaded': uploaded})

@app.route('/api/projects/<project_name>/index', methods=['POST'])
def index_project(project_name):
    project_name = secure_filename(project_name)
    project_path = os.path.join(app.config['UPLOAD_FOLDER'], project_name)
    papers_path = os.path.join(project_path, 'papers')
    index_path = os.path.join(project_path, 'index.pkl')
    
    if not os.path.exists(papers_path):
        return jsonify({'error': 'Project not found'}), 404
    
    # Temporarily update config paths
    original_papers = config.PAPERS_DIR
    original_index = config.INDEX_PATH
    
    config.PAPERS_DIR = papers_path
    config.INDEX_PATH = index_path
    
    try:
        index_papers()
        config.PAPERS_DIR = original_papers
        config.INDEX_PATH = original_index
        return jsonify({'success': True})
    except Exception as e:
        config.PAPERS_DIR = original_papers
        config.INDEX_PATH = original_index
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<project_name>/query', methods=['POST'])
def query_project(project_name):
    project_name = secure_filename(project_name)
    index_path = os.path.join(app.config['UPLOAD_FOLDER'], project_name, 'index.pkl')
    
    if not os.path.exists(index_path):
        return jsonify({'error': 'Project not indexed'}), 400
    
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 5)
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    try:
        response = rag_query(query, index_path, k=k)
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<project_name>', methods=['DELETE'])
def delete_project(project_name):
    project_name = secure_filename(project_name)
    project_path = os.path.join(app.config['UPLOAD_FOLDER'], project_name)
    
    if not os.path.exists(project_path):
        return jsonify({'error': 'Project not found'}), 404
    
    import shutil
    shutil.rmtree(project_path)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)