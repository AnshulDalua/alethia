from flask import Flask, redirect, request, session, url_for, render_template_string, flash
import requests
from github import Github
import os
import pdfplumber
import re
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from fuzzywuzzy import process
import plotly.graph_objs as go
import plotly.utils
import json
from collections import Counter
from datetime import datetime
from groq import Groq

app = Flask(__name__)
app.secret_key = 'a_secure_secret_key'

CLIENT_ID = 'cec1b2d8e1e3e6aa9caa'
CLIENT_SECRET = 'c6ba3e396f1e64dfefdd6c94ad86b51556499ef0'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fetch_file_content(file):
    if file.encoding == "base64":
        return file.path, file.decoded_content.decode('utf-8', errors='ignore')
    else:
        return file.path, None

def extract_projects_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    projects_section = re.search(r'(?i)projects(.*?)(?=Experience|Education|Skills|$)', text, re.DOTALL)
    if projects_section:
        projects_text = projects_section.group(1).strip()
        projects = re.split(r'\n(?=\w)', projects_text)
        formatted_projects = []
        for project in projects:
            lines = project.split('\n')
            formatted_project = lines[0] + "\n" + "\n".join(["• " + line.strip() for line in lines[1:] if line.strip()])
            formatted_projects.append(formatted_project.strip())
        return formatted_projects
    else:
        return []

def extract_project_names(projects):
    project_names = []
    for project in projects:
        match = re.match(r'^(.*?)\s+\|', project)
        if match:
            project_names.append(match.group(1).strip())
    return project_names

def match_repos_to_projects(project_names, repo_names):
    repo_map = {}
    for project in project_names:
        closest_match = process.extractOne(project, repo_names, score_cutoff=70)
        if closest_match:
            repo_map[project] = closest_match[0]
    return repo_map

def create_embeddings(texts):
    embeddings = embedding_model.encode(texts, convert_to_tensor=True)
    if embeddings.device.type != 'cpu':
        embeddings = embeddings.cpu()
    return embeddings

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.numpy())
    return index

def find_relevant_contexts(faiss_index, query_embedding, top_k=5):
    query_embedding = query_embedding.cpu().numpy()
    _, indices = faiss_index.search(query_embedding, top_k)
    return indices[0]

@app.route('/')
def home():
    return render_template_string('''
        <!doctype html>
        <title>Upload Resume</title>
        <h1>Upload your resume</h1>
        <form method="post" enctype="multipart/form-data" action="/upload">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
    ''')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        projects = extract_projects_from_pdf(filepath)
        session['projects'] = projects
        session['project_names'] = extract_project_names(projects)
        return redirect(url_for('login_prompt'))
    return redirect(url_for('home'))

@app.route('/login_prompt')
def login_prompt():
    return render_template_string('''
        <!doctype html>
        <title>Login with GitHub</title>
        <h1>Resume Uploaded Successfully!</h1>
        <p>Now, please login with your GitHub account to continue.</p>
        <a href="/login">Login with GitHub</a>
    ''')

@app.route('/login')
def login():
    return redirect(f'https://github.com/login/oauth/authorize?client_id={CLIENT_ID}&scope=repo')

@app.route('/callback')
def callback():
    code = request.args.get('code')
    access_token = get_access_token(code)
    session['access_token'] = access_token
    return redirect(url_for('analyze'))

def get_access_token(code):
    response = requests.post('https://github.com/login/oauth/access_token', headers={'Accept': 'application/json'}, data={
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': code
    })
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        print(f"Failed to obtain access token: {response.text}")
        return None

def fetch_repo_files(repo):
    allowed_extensions = ('.py', '.html', '.js', '.css', '.java', '.cpp', '.ts', '.tsx')
    excluded_directories = ('node_modules', 'vendor', 'dist', 'build', '__pycache__')
    files_content = []
    warnings = []
    contents = repo.get_contents("")
    while contents:
        item = contents.pop(0)
        if item.type == 'dir':
            if any(excluded in item.path for excluded in excluded_directories):
                continue
            contents.extend(repo.get_contents(item.path))
        elif item.type == 'file' and item.path.endswith(allowed_extensions):
            try:
                if item.encoding == "base64":
                    file_content = item.decoded_content.decode('utf-8', errors='ignore')
                    files_content.append((item.path, file_content))
                else:
                    warnings.append(f"Skipping file with unsupported encoding: {item.name}")
            except Exception as exc:
                warnings.append(f"Error fetching file {item.name}: {exc}")
    repo_info = {
        'name': repo.name,
        'description': repo.description,
        'language': repo.language,
        'created_at': repo.created_at,
        'updated_at': repo.updated_at,
        'size': repo.size,
        'stargazers_count': repo.stargazers_count,
        'forks_count': repo.forks_count,
    }
    return files_content, warnings, repo_info

def create_tech_stack_chart(tech_stacks):
    tech_count = Counter(tech_stacks)
    labels = list(tech_count.keys())
    values = list(tech_count.values())
    trace = go.Pie(labels=labels, values=values, textinfo='label+percent', insidetextorientation='radial')
    layout = go.Layout(title='Technology Stack Distribution')
    fig = go.Figure(data=[trace], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def analyze_code_with_llama3(file_contents, project_description, faiss_index, resume_embeddings):
    api_key = "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"
    client = Groq(api_key=api_key)
    combined_text = "\n\n".join([content for _, content in file_contents])
    if len(combined_text) > 10000:  # Assuming 10,000 characters as a safe limit
        combined_text = combined_text[:10000]
    text_embeddings = create_embeddings([combined_text])
    relevant_indices = find_relevant_contexts(faiss_index, text_embeddings)
    relevant_context = "\n".join([session['projects'][i] for i in relevant_indices])
    input_prompt = f"""
You are a strict code analyzer and resume auditor. Focus solely on analyzing the provided code for this single repository. Provide a critical evaluation based on the following aspects:

1. **Code Quality** (Score out of 10):
   - Evaluate the code's structure, readability, and adherence to best practices.
   - Provide specific examples of good or poor practices found in the code, do not give advice,.

2. **Technology Stack**:
   - List all technologies, frameworks, and tools used in the code.
   - Provide a percentage breakdown of each technology's usage in the codebase. Ensure that the percentages match to 100%
   - Compare these with the technologies claimed in the resume, highlighting any discrepancies.

3. **Project Complexity** (Score out of 10):
   - Assess the complexity of the project based on the code's functionality and structure.
   - Justify the complexity score with specific examples from the code.

4. **Test Coverage** (Score out of 10):
   - Evaluate the presence and quality of tests in the code.
   - If tests are present, assess their coverage and effectiveness.

5. **Resume Accuracy** (Score out of 10):
   - Verify the accuracy of the resume's description of the project.
   - Identify any discrepancies between the resume and the code, particularly regarding the use of specific technologies and tools.
   - Provide examples of accurate and inaccurate claims in the resume.

Relevant Context from Resume:
{relevant_context}

Project Description:
{project_description}

Code:
{combined_text}

Provide a detailed, critical analysis focusing on the above aspects. Be strict in your evaluation and highlight any discrepancies or issues found. Do not suggest improvements or discuss potential enhancements. This analysis is intended for recruiters to assess the candidate's code quality and resume accuracy.
"""
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": input_prompt}],
        model="llama-3.1-70b-versatile",
    )
    output_text = chat_completion.choices[0].message.content
    return output_text

@app.route('/analyze')
def analyze():
    github = Github(session.get('access_token'))
    user = github.get_user()
    repos = user.get_repos()
    username = user.login
    insights = []
    project_names = session.get('project_names', [])
    projects = session.get('projects', [])
    repo_names = [repo.name for repo in repos]
    repo_map = match_repos_to_projects(project_names, repo_names)
    resume_embeddings = create_embeddings(projects)
    faiss_index = build_faiss_index(resume_embeddings)

    for project, repo_name in repo_map.items():
        repo = github.get_repo(f'{username}/{repo_name}')
        project_description = next((p for p in projects if project in p), "No description found")
        try:
            files_content, warnings, repo_info = fetch_repo_files(repo)
            if warnings:
                for warning in warnings:
                    print(f"Warning: {warning}")

            # Example data, replace with actual tech stack extraction
            tech_stacks = ['Python']
            tech_stack_chart = create_tech_stack_chart(tech_stacks)

            if files_content:
                ai_insights = analyze_code_with_llama3(files_content, project_description, faiss_index, resume_embeddings)

                insights.append({
                    'repo_name': repo_info['name'],
                    'project_description': project_description,
                    'repo_info': repo_info,
                    'ai_insights': ai_insights,
                    'tech_stack_chart': tech_stack_chart
                })
            else:
                insights.append({
                    'repo_name': repo_info['name'],
                    'project_description': project_description,
                    'repo_info': repo_info,
                    'ai_insights': "No code files to analyze",
                    'tech_stack_chart': tech_stack_chart
                })
        except Exception as e:
            print(f"Error analyzing repo {repo_name}: {str(e)}")
            insights.append({
                'repo_name': repo_name,
                'project_description': project,
                'repo_info': {},
                'ai_insights': f"Error: {str(e)}",
                'tech_stack_chart': tech_stack_chart
            })

    return render_template_string('''
    <html>
    <head>
        <style>
            body {
                font-family: 'Helvetica Neue', Arial, sans-serif;
                background-color: #111;
                color: #fff;
                margin: 0;
                padding: 0;
            }
            .container {
                width: 90%;
                margin: auto;
                padding: 20px;
                max-width: 1200px;
            }
            .tab {
                overflow: hidden;
                border: 1px solid #444;
                background-color: #222;
            }
            .tab button {
                background-color: inherit;
                float: left;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 14px 16px;
                transition: 0.3s;
                color: #fff;
            }
            .tab button:hover {
                background-color: #333;
            }
            .tab button.active {
                background-color: #444;
            }
            .tabcontent {
                display: none;
                padding: 20px;
                border: 1px solid #444;
                border-top: none;
                background-color: #222;
            }
            .insight h2 {
                color: #ff4081;
                font-size: 24px;
                margin-top: 0;
            }
            .insight p, .insight ul {
                margin: 0 0 10px 0;
            }
            .insight pre {
                background-color: #333;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div class="container">
            <h1 style="color: #ff4081;">GitHub Insights Dashboard</h1>
            <div class="tab">
                {% for insight in insights %}
                    <button class="tablinks" onclick="openRepo(event, '{{ insight.repo_name }}')">{{ insight.repo_name }}</button>
                {% endfor %}
            </div>

            {% for insight in insights %}
                <div id="{{ insight.repo_name }}" class="tabcontent">
                    <div class="insight">
                        <h2>{{ insight.repo_name }}</h2>
                        <p><strong>Project Description:</strong> {{ insight.project_description }}</p>
                        <p><strong>Repository Info:</strong></p>
                        <ul>
                            <li>Language: {{ insight.repo_info.language }}</li>
                            <li>Created: {{ insight.repo_info.created_at }}</li>
                            <li>Last Updated: {{ insight.repo_info.updated_at }}</li>
                            <li>Size: {{ insight.repo_info.size }} KB</li>
                            <li>Stars: {{ insight.repo_info.stargazers_count }}</li>
                            <li>Forks: {{ insight.repo_info.forks_count }}</li>
                        </ul>
                        <div id="tech_stack_chart_{{ insight.repo_name }}" style="width:100%;height:400px;"></div>
                        <script>
                            var data = {{ insight.tech_stack_chart }};
                            Plotly.newPlot('tech_stack_chart_{{ insight.repo_name }}', data.data, data.layout);
                        </script>
                        <pre>{{ insight.ai_insights }}</pre>
                    </div>
                </div>
            {% endfor %}
        </div>
        <script>
            function openRepo(evt, repoName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(repoName).style.display = "block";
                evt.currentTarget.className += " active";
            }

            // Open the first tab by default
            document.getElementsByClassName("tablinks")[0].click();
        </script>
    </body>
    </html>
    ''', insights=insights)

if __name__ == '__main__':
    app.run(debug=True)
