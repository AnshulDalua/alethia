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

app = Flask(__name__)
app.secret_key = 'a_secure_secret_key'

CLIENT_ID = 'cec1b2d8e1e3e6aa9caa'
CLIENT_SECRET = 'c6ba3e396f1e64dfefdd6c94ad86b51556499ef0'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

OPENAI_API_KEY = 'XXX'
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
                "content": "You are a code analyzer. Please analyze the provided code and provide insights. You are to detect if an applicant's project description from their resume is an accurate representation of the code, or if it is not accurate. Please first generate a score from 1-10 regarding if the code aligns with the description. Follow this rubric: 1-3: Inaccurate, 4-6: Partially Accurate, 7-10: Accurate."
            },
            {
                "role": "user",
                "content": f"Analyze the following code and provide insights:\n\n{chunk}. See if this project description {project_description} is an accurate representation of the code, or if it is not accurate. Keep your evaluation to 3 sentences. "
            }
        ]

        response = client.chat.completions.create(
            model=MODEL, 
            messages=messages
        )

        print(response)

        insights.append(response.choices[0].message.content)
        score_match = re.search(r'\*\*Score: (\d+)\*\*', response.choices[0].message.content)
        if score_match:
            total_score += int(score_match.group(1))
    
    messages = [
            {
                "role": "system",
                "content": "You are a code analyzer. Please analyze the provided code and provide insights. You are to detect if an applicant's project description from their resume is an accurate representation of the code, or if it is not accurate. You have already analyzed each chunk of the code, and generated a score from 1-10 regarding if the code aligns with the description. You followed this rubric: 1-3: Inaccurate, 4-6: Partially Accurate, 7-10: Accurate. Now you are given all the insights for each chunk, your job is to read the analysis you have already done chunk by chunk, and give an overall score as well as insigts. Know that since you only had access to each chunk when you analyzed before, some scores may be inaccurate as the chunk you analyzed may not be directly related to the description, or maybe the chunk is less relevent part of the project. For example, you may have ranked a chunk a score of 5 when the other chunks were scores of 8-9, meaning that chunk was a non essential code, while the rest were. In this case the project should be rated a 9 as the other chunks satisfied the description. "
            },
            {
                "role": "user",
                "content": f"Analyze the following insights and provide an overall score for the applicants project to see if the project description is accurate or not:\n\n{insights}. Each insight is an isight of a chunk, your job is to read through each insight already generated and you are to provide an overall score. Know that some insights may be lower than they should, as the chunk analyzed was not a direct representation of the description, while other chunks were. Keep your evaluation to 3 sentences + score.  "
            }
        ]

    response = client.chat.completions.create(
        model=MODEL, 
        messages=messages
    )

    # average_score = total_score / num_chunks if num_chunks > 0 else 0
    # aggregated_insights = "\n\n".join(insights)
    # final_output = f"**Score: {average_score:.1f}**\n\n**Insights:**\n\n{aggregated_insights}"

    return response

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

    return files_content, warnings

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
    count = 0
    for project, repo_name in repo_map.items():
        repo = github.get_repo(f'{username}/{repo_name}')
        project_description = next((p for p in projects if project in p), "No description found")
        try:
            files_content, warnings = fetch_repo_files(repo)
            if warnings:
                for warning in warnings:
                    print(warning)
            if files_content:
                count += 1
                print(count)
                if count < 3:
                    continue
                print(project_description)

                print(repo_name)
                ai_insights = analyze_code_with_gpt4(files_content, project_description)
                insights.append(f"Repo: {repo.name}\n\nAI Insights: {ai_insights}\n\n")
            else:
                insights.append(f"Repo: {repo.name}\n\nAI Insights: No code files to analyze\n\n")
            
        except Exception as e:
            insights.append(f"Project: {project}, Repo: {repo_name}, Error: {str(e)}")
    
    insights_str = "<br><br>".join(insights)
    return f"<h1>GitHub Insights</h1><p>{insights_str}</p>"

if __name__ == '__main__':
    app.run(debug=True)
