from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import shutil
from pathlib import Path
from handle_query import rag_query, pubmed_query
from index_papers import index_papers
import config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.PROJECTS_DIR
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def is_project_indexed(project_name):
    """Check if a project has been indexed (ChromaDB directory exists)"""
    index_path = config.get_index_path(project_name)
    # Check if the ChromaDB directory exists and has the SQLite database
    chroma_db_path = os.path.join(index_path, 'chroma.sqlite3')
    return os.path.exists(chroma_db_path)


def get_projects():
    """List all project directories"""
    projects = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for d in os.listdir(app.config['UPLOAD_FOLDER']):
            project_path = config.get_project_path(d)
            if os.path.isdir(project_path):
                papers_path = config.get_papers_path(d)
                paper_count = 0
                if os.path.exists(papers_path):
                    paper_count = len([f for f in os.listdir(papers_path) if f.lower().endswith('.pdf')])
                
                indexed = is_project_indexed(d)
                projects.append({
                    'name': d,
                    'paper_count': paper_count,
                    'indexed': indexed
                })
    return projects


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/pubmed/chat', methods=['POST'])
def query_pubmed():
    data = request.json
    query = data.get('query', '')
    k = data.get('k', config.k)
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    try:
        response = pubmed_query(query, k=k)
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects', methods=['GET'])
def list_projects():
    return jsonify(get_projects())


@app.route('/api/projects', methods=['POST'])
def create_project():
    data = request.json
    project_name = secure_filename(data.get('name', ''))
    
    if not project_name:
        return jsonify({'error': 'Project name required'}), 400
    
    project_path = config.get_project_path(project_name)
    papers_path = config.get_papers_path(project_name)
    
    if os.path.exists(project_path):
        return jsonify({'error': 'Project already exists'}), 400
    
    os.makedirs(papers_path, exist_ok=True)
    return jsonify({'success': True, 'name': project_name})


@app.route('/api/projects/<project_name>/papers', methods=['GET'])
def list_papers(project_name):
    """List all papers in a project"""
    project_name = secure_filename(project_name)
    papers_path = config.get_papers_path(project_name)
    
    if not os.path.exists(papers_path):
        return jsonify({'error': 'Project not found'}), 404
    
    papers = [f for f in os.listdir(papers_path) if f.lower().endswith('.pdf')]
    return jsonify({'papers': papers})


@app.route('/api/projects/<project_name>/papers', methods=['POST'])
def upload_papers(project_name):
    project_name = secure_filename(project_name)
    papers_path = config.get_papers_path(project_name)
    
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
    """Index all papers in a project"""
    project_name = secure_filename(project_name)
    papers_path = config.get_papers_path(project_name)
    
    if not os.path.exists(papers_path):
        return jsonify({'error': 'Project not found'}), 404
    
    try:
        index_papers(project_name)
        return jsonify({'success': True, 'message': f'Successfully indexed project: {project_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<project_name>/query', methods=['POST'])
def query_project(project_name):
    """Query a project's indexed papers"""
    project_name = secure_filename(project_name)
    
    if not is_project_indexed(project_name):
        return jsonify({'error': 'Project not indexed. Please index the project first.'}), 400
    
    data = request.json
    query = data.get('query', '')
    k = data.get('k', config.k)
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    try:
        response = rag_query(query, project_name=project_name, k=k)
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<project_name>', methods=['DELETE'])
def delete_project(project_name):
    """Delete a project and all its contents"""
    project_name = secure_filename(project_name)
    project_path = config.get_project_path(project_name)
    
    if not os.path.exists(project_path):
        return jsonify({'error': 'Project not found'}), 404
    
    try:
        shutil.rmtree(project_path)
        return jsonify({'success': True, 'message': f'Project {project_name} deleted'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
