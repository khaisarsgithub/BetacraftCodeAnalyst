import datetime
import subprocess
from urllib.parse import urlencode
from venv import logger
from django.http import HttpResponse
from django.shortcuts import render, redirect
import git
from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv
import os

from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import requests
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth.models import User
from django.conf import settings

from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

from genai.views import manage_prompt_size, llm

from .prompts import base_prompt
from email_app.views import send_email, weekly_job, send_brevo_mail

from .models import Project, Report

load_dotenv()

GITHUB_CLIENT_ID = os.environ.get('GITHUB_CLIENT_ID')
GITHUB_CLIENT_SECRET = os.environ.get('GITHUB_CLIENT_SECRET')
if GITHUB_CLIENT_ID is None:
    print(f"Github Client ID is required")

# Hugging Face Transformer
encoder = SentenceTransformer("all-mpnet-base-v2")
    

# Create your views here.
def index(request):
    return redirect('github/login')

def analyze_repo_page(request):
    return render(request, 'analyze_repo.html')



# Weekly Report View
def get_weekly_report(request):
    username = request.GET.get('username')
    repo_name = request.GET.get('repo_name')
    contributor = request.GET.get('contributor')
    token = request.GET.get('token')
    emails = request.GET.get('emails')

    params = {
        'username': username,
        'repo_name': repo_name,
        'contributor': contributor,
        'token': token,
        'emails': emails
    }
    
    repo_url = f"https://{contributor}:{token}@github.com/{username}/{repo_name}.git" if token else f"https://github.com/{username}/{repo_name}.git"
    logger.info(f"Analyzing repo: {repo_url}")
    
    dest_folder = f"repositories/{username}/{repo_name}"
    clone_repo_and_get_commits(repo_url, dest_folder)
    traverse_and_copy(dest_folder, 'weekly.txt')
    report = analyze_repo(params, 'weekly.txt')

    # Store repo_name in project table
    project, created = Project.objects.get_or_create(
        name=repo_name,
        defaults={'owner': username}
    )
    # Create a new Report instance
    report = Report.objects.create(
        name=f"Weekly Report for {repo_name}",
        emails=emails,
        repository_url=f"https://github.com/{username}/{repo_name}",
        repository_token=token,
        prompt=base_prompt,
        active=True,
        frequency='Weekly',
        project=project,
        output=report
    )
    print(f"New report created for project '{repo_name}': {report}")

    if created:
        print(f"New project '{repo_name}' created for user '{username}'")
    else:
        print(f"Project '{repo_name}' already exists for user '{username}'")
    
    if not username or not repo_name:
        raise ValueError("Username and repository name are required")
    # Send Email
    try:
        today = datetime.date.today()
        last_week = today - datetime.timedelta(weeks=1)
        
        
        if emails is not None:
            send_brevo_mail(subject=f"{repo_name} : {str(last_week)[:10]} - {str(today)[:10]}", 
                html_content=report, 
                emails=emails)
            weekly_job(repo_name, report, emails)
        else:
            print("Emails not provided")
        
    except Exception as e:
        print(f"Error sending email: {e}")
        exit(1)
    return render(request, 'report.html', {'report': report})


def analyze_complete_repo(request):
    username = request.GET.get('username')
    repo_name = request.GET.get('repo_name')
    contributor = request.GET.get('contributor')
    token = request.GET.get('token')
    emails = request.GET.get('emails')
    
    params = {
        'username': username,
        'repo_name': repo_name,
        'contributor': contributor,
        'token': token,
        'emails': emails
    }
    
    if not username or not repo_name:
        raise ValueError("Username and repository name are required")
    
    repo_url = f"https://{contributor}:{token}@github.com/{username}/{repo_name}.git" if token else f"https://github.com/{username}/{repo_name}.git"
    logger.info(f"Analyzing repo: {repo_url}")
    
    dest_folder = f"repositories/{username}/{repo_name}"
    clone_repository(repo_url, dest_folder)
    traverse_and_copy(dest_folder, 'entire_repo.txt')
    report = analyze_repo(params, 'entire_repo.txt')


    # Send Email
    try:
        today = datetime.date.today()
        last_week = today - datetime.timedelta(weeks=1)
        
        if emails is not None:
            send_brevo_mail(subject=f"{repo_name} : {str(last_week)[:10]} - {str(today)[:10]}", 
                html_content=report, 
                emails=emails)
        else:
            print("Emails not provided")
                
    except Exception as e:
        print(f"Error sending email: {e}")
        exit(1)
    return render(request, 'report.html', {'report': report})
    

    
def analyze_repo(params, output_file):
    try:
        username = params['username']
        repo_name = params['repo_name']
        contributor = params['contributor']
        token = params['token']
        emails = params['emails']
        
        # if not username or not repo_name:
        #     raise ValueError("Username and repository name are required")
        
        # repo_url = f"https://{contributor}:{token}@github.com/{username}/{repo_name}.git" if token else f"https://github.com/{username}/{repo_name}.git"
        # logger.info(f"Analyzing repo: {repo_url}")

        # dest_folder = f"repositories/{username}/{repo_name}"
        # clone_repository(repo_url, dest_folder)
        # traverse_and_copy(dest_folder, 'output.txt')


        # Checkpoint Here
        data = load_data(output_file)
        documents = split_data(data)
        print("Indexing Data...")
        vectors = vectorize_data(documents)
        index = create_faiss_index(vectors)
        
        vec = encoder.encode(base_prompt).reshape(1, -1)
        D, I = index.search(vec, 4)
        context = [documents[i] for i in I[0]]
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        return JsonResponse({"error": str(e)}, status=400)
    except git.exc.GitCommandError as e:
        logger.error(f"Git error: {str(e)}")
        return JsonResponse({"error": "Failed to clone repository"}, status=500)
    except Exception as e:
        logger.exception("Unexpected error during repo analysis")
        return JsonResponse({"error": "An unexpected error occurred"}, status=500)
        
    if not context:
        print("Context Empty or Details not Provided")
        
    try:
        total_tokens = 0
        final_response = None
        # repo = request.GET.get('repo')
        prompts = manage_prompt_size(base_prompt, context)
        print(f"Number of Prompts: {len(prompts)}")
        responses = []
        print("Generating Response...")
        output_file = "output.txt"
        for prompt in prompts:
            print(f"Prompt: {llm.count_tokens(prompt)}")
            response = llm.generate_content(prompt)
            report = response.text
            print(f"Output: {llm.count_tokens(response.text)}")
            print(response.text)
            responses.append(response.text)
        # if len(responses) > 1:
        return response.text
    
    except Exception as e:
        logger.exception("Unexpected error during repo analysis")
        return JsonResponse({"error": "An unexpected error occurred"}, status=500)

def clone_repo_and_get_commits(repo_url, dest_folder):
    content = ""
    # dest_folder = "./repo/" + repo_url.split('/')[-1].replace('.git', '')
    if not os.path.exists(dest_folder):
        try:
            git.Repo.clone_from(repo_url, dest_folder)
            print(f"Repository cloned to {dest_folder}")
        except git.exc.GitCommandError as e:
            print(f"Error cloning repository: {e}")
            content = f"Error cloning repository: {e}"
    else:
        print(f"Repository already cloned to {dest_folder}")

    # Initialize GitPython Repo object
    print(f"Initializing Repo : {dest_folder}")
    repo = git.Repo(dest_folder)

    # Get the commits from the last week
    last_week = datetime.datetime.now() - datetime.timedelta(weeks=1)
    commits = list(repo.iter_commits(since=last_week.isoformat()))

    # Print the commit details
    if not commits:
        print("No commits found in the last week")
        content = "<h2>No commits found in the last week</h2>"
    else:
        content = commit_diff(commits)
    return content


    

def commit_diff(commits):
    content = ""
    for commit in commits:
        print("Commmit")
        print(f"Commit: {commit.hexsha}")
        print(f"Author: {commit.author.name}")
        print(f"Date: {commit.committed_datetime}")
        print(f"Message: {commit.message}")
        print("\n" + "-"*60 + "\n")
        # print("Changes:")
        
        # Iterate over all files in the commit
        for item in commit.tree.traverse():
            if isinstance(item, git.objects.blob.Blob):
                file_path = item.path
                blob = commit.tree / file_path
                file_contents = blob.data_stream.read()
                content += f"\n\n--- {file_path} ---\n\n"
                content += f"```{file_contents}```"


        # Parent commits
        parent_shas = [parent.hexsha for parent in commit.parents]
        print(f"Parent Commits: {', '.join(parent_shas)}")
        content += f"Parent Commits: {', '.join(parent_shas)} <br>"
        # Commit stats
        stats = commit.stats.total
        # content += str(stats)
        print(f"Stats: {stats}")
        # commits_changes = f"""Commit: {commit.hexsha}\n Author: {commit.author.name}\nDate: {commit.committed_datetime}\nMessage: {commit.message}\n
                # Parent Commits: {', '.join(parent_shas)}\nStats: {stats}"""

        # Diff with parent
        if commit.parents:
            diffs = commit.diff(commit.parents[0])
            for diff in diffs:
                content += f"<br> Changed Files: <br> --- {diff.a_path} ---"
                print("Difference:")
                print(f"File: {diff.a_path}")
                print(f"New file: {diff.new_file}")
                print(f"Deleted file: {diff.deleted_file}")
                print(f"Renamed file: {diff.renamed_file}")
                # print(f"Changes:\n{diff.diff}")

                if diff.diff:
                    print(diff.diff.decode('utf-8'))#, language='diff')

            print("\n" + "-"*60 + "\n")
        # print(f"Content: \n{content}")
    with open('output.txt', 'w') as f:
        f.write(content)
    return content
    
# Function to clone Repository
def clone_repository(repo_url, dest_folder):
    print(f"Repo URL in Clone Repository: {repo_url}")
    if not os.path.exists(dest_folder):
        try:
            subprocess.run(['git', 'clone', repo_url, dest_folder])
        except Exception as e:
            print(f"Error cloning repository: {e}")
            exit(1)
    else: print("Repository already exists")

# Function to traverse all files and write their contents to a single file
def traverse_and_copy(src_folder, output_file):
    # Define unwanted file extensions or patterns
    unwanted_extensions = [
        '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.exe', '.bin', 
        '.lock', '.generators', '.yml', '.scss', '.css', '.html', '.erb',
        '.sample', '.rake']
    unwanted_files = ['LICENSE', 'README.md', '.dockerignore',  'manifest.js', 'exclude']
    print("Copying the files")
    print(f"Skipping Extensions {unwanted_extensions} and Files {unwanted_files}.")
    with open(output_file, 'w', encoding='utf-8', errors='ignore') as outfile:
        for root, _, files in os.walk(src_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if ((os.path.splitext(file)[1].lower() in unwanted_extensions) or 
                    (file in unwanted_files) or 
                    (is_binary(file_path))):
                    continue
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                    outfile.write(f"--- {file_path} ---\n")
                    outfile.write(infile.read())
                    outfile.write("\n\n")
                
# Function to check if the file is binary
def is_binary(file_path):
    try:
        with open(file_path, 'rb') as file:
            for chunk in iter(lambda: file.read(1024), b''):
                if b'\0' in chunk:
                    return True
    except Exception as e:
        print(f"Could not read {file_path} to check if it's binary: {e}")
    return False

# Function to load data from file
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")
    loader = TextLoader(file_path, encoding='utf-8')
    data = loader.load()
    return data

def split_data(data, chunk_size=999500):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, separators=['\n\n'])
    splits = text_splitter.split_documents(data)
    documents = [split.page_content for split in splits]
    return documents

# Function to vectorize data
def vectorize_data(documents):
    vectors = encoder.encode(documents)
    return vectors

# Function to create or update FAISS index
def create_faiss_index(vectors, index=None):
    dim = vectors.shape[1]
    if index is None:
        index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index



def github_login(request):
    if not GITHUB_CLIENT_ID:
        return HttpResponse("Github Client ID is required", status=400)
    
    github = OAuth2Session(
        client_id=GITHUB_CLIENT_ID,
        redirect_uri='https://localhost:8000/github/callback',
    )
    authorization_url, state = github.authorization_url('https://github.com/login/oauth/authorize')
    request.session['oauth_state'] = state
    print(f"Login Request Object: {request}")
    return redirect(authorization_url)

def github_callback(request):
    try:
        github = OAuth2Session(
            client_id=GITHUB_CLIENT_ID,
            redirect_uri='https://localhost:8000/github/callback',
            state=request.session.get('oauth_state')
        )
        token = github.fetch_token(
            'https://github.com/login/oauth/access_token',
            client_secret=GITHUB_CLIENT_SECRET,
            authorization_response=request.build_absolute_uri()
        )

        # Print token for debugging purposes
        print(f"Github Token: {token}")

        # Use the token to access the GitHub API and get the user's information
        response = requests.get(
            'https://api.github.com/user',
            headers={'Authorization': f'token {token["access_token"]}'}
        )

        # Ensure the request was successful
        if response.status_code == 200:
            user_data = response.json()
            print(f"User Data: {user_data}")
            github_username = user_data['login']
            print(f"Github Data: {user_data}")

            # Optionally, you can create a Django user or use an existing one
            user, _ = User.objects.get_or_create(username=github_username)

            # Generate JWT tokens (both access and refresh)
            refresh = RefreshToken.for_user(user)
            access_token = str(refresh.access_token)

            # Customize the JWT payload (adding GitHub token)
            refresh['github_token'] = token['access_token']

            print("Fetching User Repositories...")
            git_repositories = get_github_repos(github_username, token['access_token'])
            print(f"Repositories : {git_repositories}")

            return render(request, 'index.html', {'repositories':git_repositories, 
                                                  'username':github_username, 
                                                  'token': token['access_token']})

        else:
            return handle_error("Failed to retrieve GitHub user information")

    except Exception as e:
        print(f"Error during GitHub OAuth: {e}")
        return handle_error(f"An error occurred: {e}")


def get_github_repos(username, token):
    # username = request.GET.get('username')
    # token = request.GET.get('token')

    url = f"https://api.github.com/users/{username}/repos"
    headers = {
        'Authorization': f'token {token}'
    }
    params = {
        'visibility': 'all'  # 'all' to include both public and private repositories
    }

    # Make the request to the GitHub API
    response = requests.get(url, headers=headers, params=params)

    print(f"Response: {response.status_code}")
    # Check if the request was successful
    if response.status_code == 200:
        try:
            repos = response.json()
        except ValueError:
            print("Failed to parse response as JSON.")
            return {"error": "Failed to parse response as JSON."}
    
        # List to hold repo details
        repo_details = []
        # print(f"Repos: {repos}")

        # Iterate through the repositories
        for repo in repos:
            # Basic repo information
            repo_name = repo['name']
            repo_url = repo['html_url']
            repo_description = repo['description']
            repo_language = repo['language']
            repo_stars = repo['stargazers_count']
            repo_forks = repo['forks_count']
            repo_watchers = repo['watchers_count']
            repo_open_issues = repo['open_issues_count']

            # Fetch pull requests
            try:
                pulls_url = repo['pulls_url'].replace("{/number}", "")
                pulls_response = requests.get(pulls_url)
                pulls = pulls_response.json() if pulls_response.status_code == 200 else []
            except Exception as e:
                print(f"Error: While getting Pulls: {e}")

            # Fetch commits
            try:
                commits_url = repo['commits_url'].replace("{/sha}", "")
                commits_response = requests.get(commits_url)
                commits = commits_response.json() if commits_response.status_code == 200 else []
            except Exception as e:
                print(f"Error: While getting Commits: {e}")
            
            # Append the gathered information to the list
            repo_details.append({
                'name': repo_name,
                'url': repo_url,
                'description': repo_description,
                'language': repo_language,
                'stars': repo_stars,
                'forks': repo_forks,
                'watchers': repo_watchers,
                'open_issues': repo_open_issues,
                'pull_requests': len(pulls),
                'commits': len(commits)
            })

        return repo_details
    else:
        return f"Failed to fetch repositories for user {username}. HTTP Status code: {response.status_code}"


def handle_error(error_message):
    # Handle the error appropriately, you could redirect to an error page or return an error response
    return HttpResponse(f"Error: {error_message}", status=400)