# from flask import Flask, redirect, request, session, url_for
# import requests
# from github import Github
# import os
# from getpass import getpass
# from llama_index.core import Settings, VectorStoreIndex
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.replicate import Replicate
# from tavily import TavilyClient
# from llama_index.core import Document

# # Set up Flask application
# app = Flask(__name__)
# app.secret_key = 'a_secure_secret_key'

# # Set up API credentials
# CLIENT_ID = 'cec1b2d8e1e3e6aa9caa'
# CLIENT_SECRET = 'c6ba3e396f1e64dfefdd6c94ad86b51556499ef0'

# REPLICATE_API_TOKEN = 'r8_YhWYKlpbaj44es5GXl7q54yeOoFYN464HIqmq'
# os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# TAVILY_API_KEY = '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'
# tavily = TavilyClient(api_key=TAVILY_API_KEY)

# Settings.llm = Replicate(
#     model="meta/meta-llama-3-8b-instruct",
#     temperature=0.0,
#     additional_kwargs={"top_p": 1, "max_new_tokens": 500},
# )

# Settings.embed_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-small-en-v1.5"
# )

# @app.route('/')
# def home():
#     return '<a href="/login">Login with GitHub</a>'

# @app.route('/login')
# def login():
#     return redirect(f'https://github.com/login/oauth/authorize?client_id={CLIENT_ID}&scope=repo')

# @app.route('/callback')
# def callback():
#     code = request.args.get('code')
#     access_token = get_access_token(code)
#     session['access_token'] = access_token
#     return redirect('/analyze')

# def get_access_token(code):
#     response = requests.post('https://github.com/login/oauth/access_token', headers={'Accept': 'application/json'}, data={
#         'client_id': CLIENT_ID,
#         'client_secret': CLIENT_SECRET,
#         'code': code
#     })
#     if response.status_code == 200:
#         return response.json().get('access_token')
#     else:
#         print(f"Failed to obtain access token: {response.text}")
#         return None

# def analyze_with_llama(description):
#     # Fetch Llama 3 insights
#     response = tavily.search(query=description)
#     context = [{"url": obj["url"], "content": obj["content"]} for obj in response['results']]
#     documents = [Document(text=ct['content']) for ct in context]
#     index = VectorStoreIndex.from_documents(documents)

#     query_engine = index.as_query_engine(streaming=True)

#     return query_engine.query(f"Analyze: {description}")

# @app.route('/analyze')
# def analyze():
#     github = Github(session.get('access_token'))
#     user = github.get_user()
#     repos = user.get_repos()
#     insights = []
#     for repo in repos:
#         description = repo.description if repo.description else "No description available."
#         ai_insights = analyze_with_llama(description)
#         insights.append(f'Repo: {repo.name}, AI Insights: {ai_insights}')
#     return '<br>'.join(insights)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, redirect, request, session, url_for
import requests
from github import Github
import os
from transformers import pipeline
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
import replicate
from tavily import TavilyClient
from langchain.document_loaders import GithubFileLoader

# Set up Flask application
app = Flask(__name__)
app.secret_key = 'a_secure_secret_key'

# Set up API credentials
CLIENT_ID = 'cec1b2d8e1e3e6aa9caa'
CLIENT_SECRET = 'c6ba3e396f1e64dfefdd6c94ad86b51556499ef0'

# REPLICATE_API_TOKEN = 'r8_YhWYKlpbaj44es5GXl7q54yeOoFYN464HIqmq'
REPLICATE_API_TOKEN= 'r8_QfhQKhuNvwF2uZUz8UlaT8Nv9LAqWu019ddDh'
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
TAVILY_API_KEY = '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'
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# Initialize AI models
# Settings.llm = Replicate(
#     model="meta/meta-llama-3-8b-instruct",
#     temperature=0.0,
#     additional_kwargs={"top_p": 1, "max_new_tokens": 500},
# )
# Settings.embed_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-small-en-v1.5"
# )

# Text and code analysis models
# text_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# code_summarizer = pipeline("text-generation", model="Salesforce/codet5-base-multi-sum")

@app.route('/')
def home():
    return '<a href="/login">Login with GitHub</a>'

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

# @app.route('/analyze')
# def analyze():
#     github = Github(session.get('access_token'))
#     user = github.get_user()
#     repos = user.get_repos()
#     insights = []
#     for repo in repos:
#         try:
#             # Fetch all files recursively
#             contents = repo.get_contents("")
#             repo_insights = []
#             while contents:
#                 file_content = contents.pop(0)
#                 if file_content.type == "dir":
#                     contents.extend(repo.get_contents(file_content.path))
#                 else:
#                     if file_content.name.lower().endswith(('.py', '.js', '.java')):
#                         code = file_content.decoded_content.decode()
#                         analysis_result = code_summarizer(code, max_length=150)
#                         repo_insights.append(f"Code Summary: {analysis_result[0]['generated_text']}")
#                     elif file_content.name.lower().endswith('.md'):
#                         text = file_content.decoded_content.decode()
#                         analysis_result = text_summarizer(text, max_length=150)
#                         repo_insights.append(f"Doc Summary: {analysis_result[0]['summary_text']}")
#             insights.append(f"Repo: {repo.name}, Insights: " + ' | '.join(repo_insights))
#         except Exception as e:
#             insights.append(f"Repo: {repo.name}, Error: {str(e)}")
#     return '<br>'.join(insights)

def analyze_with_llama(repo_name,description):
    # Fetch related contexts using Tavily
    # response = tavily.search(query='https://github.com/AnshulDalua')
    # context = [{"url": obj["url"], "content": obj["content"]} for obj in response['results']]
    print(repo_name)
    loader = GithubFileLoader(
    repo=repo_name,  # the repo name
    access_token="cec1b2d8e1e3e6aa9caa",
    github_api_url="https://api.github.com",
    file_filter=lambda file_path: file_path.endswith(
        'txt', 'md', 'markdown', 'rst', 'py', 'js', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php', 'scala', 'html', 'htm', 'xml', 'json', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'sh', 'bash', 'css', 'scss', 'sql', 'gitignore', 'dockerignore', 'editorconfig', 'ipynb'
    ), 
    )
    documents = loader.load()
    print(documents)

    return "hi"


    
    # # Create documents from the fetched content
    # documents = [Document(text=ct['content']) for ct in context]
    
    # # Index documents using VectorStoreIndex
    # index = VectorStoreIndex.from_documents(documents)
    
    # # Query the index for insights related to the description
    # query_engine = index.as_query_engine(streaming=True)
    # insights = query_engine.query(f"Analyze this github repo: ")
    # The meta/meta-llama-3-8b-instruct model can stream output as it's running.
    # for event in replicate.stream(
    #     "meta/meta-llama-3-8b-instruct",
    #     input={
    #         "top_k": 0,
    #         "top_p": 0.9,
    #         "prompt": "Write me three poems about llamas, the first in AABB format, the second in ABAB, the third without any rhyming",
    #         "temperature": 0.6,
    #         "system_prompt": "You are a helpful assistant",
    #         "length_penalty": 1,
    #         "max_new_tokens": 512,
    #         "stop_sequences": "<|end_of_text|>,<|eot_id|>",
    #         "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    #         "presence_penalty": 0
    #     },
        
    # ):

    # input = {
    #     "prompt": "analyze this repository: https://github.com/AnshulDalua",
    #     "prompt_template": "Analyze this code. Access the URL. Give insights on the code.",
    #     "presence_penalty": 0,
    #     "frequency_penalty": 0
    # }

    # output = replicate.run(
    #     "meta/meta-llama-3-8b-instruct",
    #     input=input
    # )



    return "hi"

@app.route('/analyze')
def analyze():
    github = Github(session.get('access_token'))
    user = github.get_user()
    repos = user.get_repos()
    print(repos)
    print(user)
    
    insights = []
    for repo in repos:
        description = repo.description if repo.description else "No description available."
        try:
            # Use Llama to analyze repository description or other relevant content
            ai_insights = analyze_with_llama(repo.name, description)
            insights.append(f'Repo: {repo.name}, AI Insights: {ai_insights}')
        except Exception as e:
            insights.append(f"Repo: {repo.name}, Error: {str(e)}")
    return '<br>'.join(insights)

    # github = Github(session.get('access_token'))
    # user = github.get_user()
    # repos = user.get_repos()
    # insights = []
    # for repo in repos:
    #     files_content = fetch_repo_files(github, repo)
    #     for file, content in files_content.items():
    #         try:
    #             result = analyze_with_llama(content)
    #             insights.append(f"File: {file}, AI Insights: {result[0]['generated_text']}")
    #         except Exception as e:
    #             insights.append(f"File: {file}, Error: {str(e)}")
    # return '<br>'.join(insights)

if __name__ == '__main__':
    app.run(debug=True)
