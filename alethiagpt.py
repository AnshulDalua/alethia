from flask import Flask, redirect, request, session, url_for, render_template_string, flash
import requests
from github import Github
import os
import pdfplumber
import re
from werkzeug.utils import secure_filename
from langchain.document_loaders import GithubFileLoader
from langchain.text_splitter import CharacterTextSplitter
import logging
from fuzzywuzzy import process
from openai import OpenAI
import plotly.graph_objs as go
import plotly.utils
import json
from collections import Counter
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'a_secure_secret_key'

CLIENT_ID = 'cec1b2d8e1e3e6aa9caa'
CLIENT_SECRET = 'c6ba3e396f1e64dfefdd6c94ad86b51556499ef0'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

OPENAI_API_KEY = 'sk-proj-Crh7jg2GxphwO9yThoj8T3BlbkFJ2GM7V8dVPqoFoM8CYZiK'
client = OpenAI(
    api_key=OPENAI_API_KEY,
)
MODEL = 'gpt-4o'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fetch_file_content(file):
    """Fetches content of a single file."""
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

def split_text_into_chunks(text, max_tokens=2000):
    max_length = max_tokens * 4
    chunks = []

    while len(text) > max_length:
        split_index = text.rfind('\n', 0, max_length)
        if split_index == -1:
            split_index = max_length
        
        chunks.append(text[:split_index])
        text = text[split_index:]

    chunks.append(text)
    return chunks

def analyze_code_with_gpt4(file_contents, project_description):
    combined_text = "\n\n".join([content for _, content in file_contents])
    chunks = split_text_into_chunks(combined_text)
    insights = []
    total_score = 0
    num_chunks = len(chunks)
    for chunk in chunks:
        messages = [
            {
                "role": "system",
                "content": "You are a code analyzer. Analyze the provided code and provide insights on code quality, technology stack, project complexity, and test coverage. Also, detect if the project description from the resume is accurate."
            },
            {
                "role": "user",
                "content": f"Analyze the following code and provide insights:\n\n{chunk}\n\nProject description: {project_description}\n\nProvide insights on:\n1. Code quality (1-10 score)\n2. Technology stack used\n3. Project complexity (1-10 score)\n4. Test coverage (if applicable)\n5. Accuracy of project description (1-10 score)\n\nKeep your evaluation concise and relevant for recruiters."
            }
        ]

        response = client.chat.completions.create(
            model=MODEL, 
            messages=messages
        )

        insights.append(response.choices[0].message.content)
        scores = re.findall(r'(\d+)/10', response.choices[0].message.content)
        if scores:
            total_score += sum(map(int, scores))
    
    # Aggregate insights
    messages = [
        {
            "role": "system",
            "content": "You are a code analyzer. Summarize the insights from multiple code chunks and provide an overall assessment."
        },
        {
            "role": "user",
            "content": f"Summarize the following insights and provide an overall assessment:\n\n{insights}\n\nProvide a concise summary focusing on:\n1. Overall code quality\n2. Main technologies used\n3. Project complexity\n4. Test coverage\n5. Accuracy of project description\n\nKeep your summary to 5-7 sentences, tailored for recruiters."
        }
    ]

    response = client.chat.completions.create(
        model=MODEL, 
        messages=messages
    )

    return response.choices[0].message.content

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

    # Get additional repository information
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

def create_code_quality_chart(repo_names, code_quality_scores):
    trace = go.Bar(x=repo_names, y=code_quality_scores)
    layout = go.Layout(title='Code Quality Scores by Repository', xaxis_title='Repository', yaxis_title='Code Quality Score')
    fig = go.Figure(data=[trace], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_project_complexity_chart(repo_names, complexity_scores, creation_dates):
    trace = go.Scatter(x=creation_dates, y=complexity_scores, mode='markers', text=repo_names, marker=dict(size=10))
    layout = go.Layout(title='Project Complexity Over Time', xaxis_title='Creation Date', yaxis_title='Complexity Score')
    fig = go.Figure(data=[trace], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

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
    
    tech_stacks = []
    code_quality_scores = []
    complexity_scores = []
    creation_dates = []
    
    for project, repo_name in repo_map.items():
        repo = github.get_repo(f'{username}/{repo_name}')
        project_description = next((p for p in projects if project in p), "No description found")
        try:
            files_content, warnings, repo_info = fetch_repo_files(repo)
            if warnings:
                for warning in warnings:
                    print(warning)
            if files_content:
                ai_insights = analyze_code_with_gpt4(files_content, project_description)
                
                # Extract technology stack, code quality, and complexity scores
                tech_stack_match = re.search(r'Main technologies used: (.*)', ai_insights)
                if tech_stack_match:
                    tech_stacks.extend([tech.strip() for tech in tech_stack_match.group(1).split(',')])
                
                code_quality_match = re.search(r'Overall code quality: (\d+)/10', ai_insights)
                if code_quality_match:
                    code_quality_scores.append((repo_name, int(code_quality_match.group(1))))
                
                complexity_match = re.search(r'Project complexity: (\d+)/10', ai_insights)
                if complexity_match:
                    complexity_scores.append((repo_name, int(complexity_match.group(1))))
                    creation_dates.append(repo_info['created_at'])
                
                insights.append(f"""
                <div class="insight">
                    <h2>Repo: {repo_info['name']}</h2>
                    <p><strong>Project Description:</strong> {project_description}</p>
                    <p><strong>Repository Info:</strong></p>
                    <ul>
                        <li>Language: {repo_info['language']}</li>
                        <li>Created: {repo_info['created_at']}</li>
                        <li>Last Updated: {repo_info['updated_at']}</li>
                        <li>Size: {repo_info['size']} KB</li>
                        <li>Stars: {repo_info['stargazers_count']}</li>
                        <li>Forks: {repo_info['forks_count']}</li>
                    </ul>
                    <pre>{ai_insights}</pre>
                </div>
                """)
            else:
                insights.append(f"""
                <div class="insight">
                    <h2>Repo: {repo_info['name']}</h2>
                    <p>No code files to analyze</p>
                </div>
                """)
        except Exception as e:
            insights.append(f"""
            <div class="insight">
                <h2>Project: {project}, Repo: {repo_name}</h2>
                <p>Error: {str(e)}</p>
            </div>
            """)
    
    # Create charts
    tech_stack_chart = create_tech_stack_chart(tech_stacks)
    code_quality_chart = create_code_quality_chart([score[0] for score in code_quality_scores], [score[1] for score in code_quality_scores])
    project_complexity_chart = create_project_complexity_chart([score[0] for score in complexity_scores], [score[1] for score in complexity_scores], creation_dates)
    
    insights_str = "".join(insights)
    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
            }}
            .container {{
                width: 80%;
                margin: auto;
            }}
            .insight {{
                margin-bottom: 20px;
            }}
            .insight h2 {{
                background-color: #f2f2f2;
                padding: 10px;
                border-left: 5px solid #333;
            }}
            .insight p {{
                padding: 10px;
                background-color: #f9f9f9;
            }}
            .insight pre {{
                background-color: #eef;
                padding: 10px;
                border: 1px solid #ddd;
                white-space: pre-wrap;
            }}
            .chart {{
                width: 100%;
                height: 400px;
                margin-bottom: 20px;
            }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>GitHub Insights</h1>
            <div id="tech-stack-chart" class="chart"></div>
            <div id="code-quality-chart" class="chart"></div>
            <div id="project-complexity-chart" class="chart"></div>
            {insights_str}
        </div>
        <script>
            var techStackData = {tech_stack_chart};
            Plotly.newPlot('tech-stack-chart', techStackData.data, techStackData.layout);
            
            var codeQualityData = {code_quality_chart};
            Plotly.newPlot('code-quality-chart', codeQualityData.data, codeQualityData.layout);
            
            var projectComplexityData = {project_complexity_chart};
            Plotly.newPlot('project-complexity-chart', projectComplexityData.data, projectComplexityData.layout);
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)