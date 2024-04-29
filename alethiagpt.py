from flask import Flask, redirect, request, session, url_for
import requests
from github import Github
import os
from getpass import getpass
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
from tavily import TavilyClient
from llama_index.core import Document

# Set up Flask application
app = Flask(__name__)
app.secret_key = 'a_secure_secret_key'

# Set up API credentials
CLIENT_ID = 'cec1b2d8e1e3e6aa9caa'
CLIENT_SECRET = 'c6ba3e396f1e64dfefdd6c94ad86b51556499ef0'

REPLICATE_API_TOKEN = 'r8_YhWYKlpbaj44es5GXl7q54yeOoFYN464HIqmq'
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

TAVILY_API_KEY = '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'
tavily = TavilyClient(api_key=TAVILY_API_KEY)

Settings.llm = Replicate(
    model="meta/meta-llama-3-8b-instruct",
    temperature=0.0,
    additional_kwargs={"top_p": 1, "max_new_tokens": 500},
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

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
    return redirect('/analyze')

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

def analyze_with_llama(description):
    # Fetch Llama 3 insights
    response = tavily.search(query=description)
    context = [{"url": obj["url"], "content": obj["content"]} for obj in response['results']]
    documents = [Document(text=ct['content']) for ct in context]
    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine(streaming=True)

    return query_engine.query(f"Analyze: {description}")

@app.route('/analyze')
def analyze():
    github = Github(session.get('access_token'))
    user = github.get_user()
    repos = user.get_repos()
    insights = []
    for repo in repos:
        description = repo.description if repo.description else "No description available."
        ai_insights = analyze_with_llama(description)
        insights.append(f'Repo: {repo.name}, AI Insights: {ai_insights}')
    return '<br>'.join(insights)

if __name__ == '__main__':
    app.run(debug=True)
