import os
import io
import re
import zipfile
import requests
from urllib.parse import quote
from flask import Flask, render_template, request
from google import genai
from google.genai import types

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload limit


# Extensions supported by the reviewer.  Keeping this mapping in one place makes
# it easy to add a language without changing the upload and GitHub workflows.
LANGUAGE_BY_EXTENSION = {
    '.py': 'Python',
    '.pyw': 'Python',
    '.js': 'JavaScript',
    '.mjs': 'JavaScript',
    '.cjs': 'JavaScript',
    '.jsx': 'JavaScript (JSX)',
    '.ts': 'TypeScript',
    '.tsx': 'TypeScript (TSX)',
    '.c': 'C',
    '.h': 'C/C++ header',
    '.cc': 'C++',
    '.cpp': 'C++',
    '.cxx': 'C++',
    '.hpp': 'C++ header',
    '.hh': 'C++ header',
    '.hxx': 'C++ header',
    '.sql': 'SQL',
}


def language_for_filename(filename: str) -> str:
    """Return the programming language inferred from a file name."""
    extension = os.path.splitext(filename.lower())[1]
    return LANGUAGE_BY_EXTENSION.get(extension, 'Unknown')


def is_supported_source_file(filename: str) -> bool:
    return language_for_filename(filename) != 'Unknown'


def detect_language_from_code(code: str) -> str:
    """Best-effort language detection for code pasted without a filename."""
    normalized = code.strip()
    if re.search(r'^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|WITH)\b', normalized, re.I):
        return 'SQL'
    if re.search(r'^\s*(#include\s*[<"]|using\s+namespace\s+|std::)', normalized, re.M):
        return 'C/C++'
    if re.search(r'^\s*(def |class |import |from .+ import |async def )', normalized, re.M):
        return 'Python'
    if re.search(r'\b(function|const|let|var|export|import)\b|=>', normalized):
        return 'JavaScript/TypeScript'
    return 'the detected source language'


class CleanCodeReviewer:
    """Multi-Agent reviewer handling Educational, Quick Fix, and Repository Audit modes."""

    def __init__(self, model_name: str = "gemini-3.6-flash"):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.model_name = model_name

    def review_educational(self, code: str, language: str = 'the detected source language') -> str:
        """Mode 1: Full architectural breakdown and explanation."""
        system_instruction = (
            f"You are a Senior {language} Software Architect and strict Clean Code reviewer. "
            "Analyze the code and evaluate based on naming, SRP, magic numbers, function sizes, and error handling.\n"
            "Format response as:\n"
            "1. VERDICT: [CLEAN] or [NEEDS REFACTORING]\n"
            "2. ISSUES FOUND: Bullet points of specific violations.\n"
            "3. RECOMMENDATIONS: How to fix them.\n"
            f"4. REFACTORED CODE: Provide full clean code in a {language} code block."
        )
        return self._generate(code, system_instruction, temperature=0.2)

    def review_quick_fix(self, code: str, language: str = 'the detected source language') -> str:
        """Mode 2: Returns ONLY the refactored code without extra conversational text."""
        system_instruction = (
            f"You are an automated code formatter. Convert the provided {language} code into Clean Code. "
            "Do NOT include explanations, verdicts, or intro text. "
            f"Output ONLY valid, cleaned {language} code in a fenced code block."
        )
        return self._generate(code, system_instruction, temperature=0.1)

    def review_github_file(self, filename: str, code: str) -> dict:
        """Mode 3 helper: Audits individual repo files."""
        system_instruction = (
            f"Analyze the provided {language_for_filename(filename)} file for Clean Code principles. "
            "Respond in JSON format with keys:\n"
            "- 'is_clean': boolean\n"
            "- 'summary': brief sentence of findings\n"
            "- 'refactored_code': string with clean python code if not clean, else original"
        )
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=f"File: {filename}\n\nCode:\n{code}",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            import json
            return json.loads(response.text)
        except Exception as e:
            return {"is_clean": True, "summary": f"Error auditing file: {str(e)}", "refactored_code": code}

    def _generate(self, prompt: str, system_instruction: str, temperature: float) -> str:
        if not self.client:
            return "Error: GEMINI_API_KEY is missing from environment variables."
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature
                )
            )
            return response.text
        except Exception as e:
            return f"Error during analysis: {str(e)}"


def extract_source_code_from_upload(file_storage) -> str:
    """Extract supported source files from an upload or a ZIP archive."""
    filename = file_storage.filename.lower()
    
    if is_supported_source_file(filename):
        return file_storage.read().decode('utf-8', errors='ignore')
        
    elif filename.endswith('.zip'):
        extracted_code = []
        with zipfile.ZipFile(io.BytesIO(file_storage.read())) as z:
            for zip_info in z.infolist():
                if (not zip_info.is_dir() and not zip_info.filename.startswith('__MACOSX')
                        and is_supported_source_file(zip_info.filename)):
                    with z.open(zip_info) as f:
                        code_content = f.read().decode('utf-8', errors='ignore')
                        language = language_for_filename(zip_info.filename)
                        extracted_code.append(
                            f"// --- File: {zip_info.filename} ({language}) ---\n{code_content}\n"
                        )
        return "\n".join(extracted_code) if extracted_code else "No supported source files found in ZIP archive."
    
    return ""


def fetch_github_source_files(repo_url: str) -> dict:
    """Fetch supported source files from a public GitHub repository."""
    match = re.search(r"github\.com/([^/]+)/([^/.]+)", repo_url)
    if not match:
        return {"error": "Invalid GitHub repository URL format."}

    owner, repo = match.group(1), match.group(2)
    repo_response = requests.get(f"https://api.github.com/repos/{owner}/{repo}", timeout=15)
    if repo_response.status_code != 200:
        return {"error": f"Failed to fetch repository. Status code: {repo_response.status_code}"}

    default_branch = repo_response.json().get('default_branch', 'main')
    api_url = (
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/"
        f"{quote(default_branch, safe='')}?recursive=1"
    )
    res = requests.get(api_url, timeout=15)
    if res.status_code != 200:
        return {"error": f"Failed to fetch repository. Status code: {res.status_code}"}

    tree = res.json().get('tree', [])
    source_files = {}

    for item in tree:
        if item['type'] == 'blob' and is_supported_source_file(item['path']):
            raw_url = (
                f"https://raw.githubusercontent.com/{owner}/{repo}/"
                f"{quote(default_branch, safe='')}/{quote(item['path'], safe='/')}"
            )
            file_res = requests.get(raw_url, timeout=15)
            if file_res.status_code == 200:
                source_files[item['path']] = file_res.text

    return {"files": source_files, "repo_name": f"{owner}/{repo}"}


@app.route('/', methods=['GET', 'POST'])
def index():
    active_tab = request.form.get('active_tab', 'edu')
    dirty_code = ""
    feedback = None
    github_results = None
    error = None

    if request.method == 'POST':
        reviewer = CleanCodeReviewer()

        # MODE 1: Educational
        if active_tab == 'edu':
            if 'file_upload' in request.files and request.files['file_upload'].filename != '':
                uploaded_file = request.files['file_upload']
                dirty_code = extract_source_code_from_upload(uploaded_file)
                language = (
                    'multiple source languages' if uploaded_file.filename.lower().endswith('.zip')
                    else language_for_filename(uploaded_file.filename)
                )
            else:
                dirty_code = request.form.get('dirty_code', '')
                language = detect_language_from_code(dirty_code)
            feedback = reviewer.review_educational(dirty_code, language)

        # MODE 2: Quick Fix
        elif active_tab == 'quick':
            if 'file_upload_quick' in request.files and request.files['file_upload_quick'].filename != '':
                uploaded_file = request.files['file_upload_quick']
                dirty_code = extract_source_code_from_upload(uploaded_file)
                language = (
                    'multiple source languages' if uploaded_file.filename.lower().endswith('.zip')
                    else language_for_filename(uploaded_file.filename)
                )
            else:
                dirty_code = request.form.get('dirty_code_quick', '')
                language = detect_language_from_code(dirty_code)
            feedback = reviewer.review_quick_fix(dirty_code, language)

        # MODE 3: GitHub Repo Auditor
        elif active_tab == 'github':
            repo_url = request.form.get('repo_url', '')
            repo_data = fetch_github_source_files(repo_url)

            if "error" in repo_data:
                error = repo_data["error"]
            else:
                github_results = []
                for filepath, code_content in repo_data["files"].items():
                    audit = reviewer.review_github_file(filepath, code_content)
                    github_results.append({
                        "file": filepath,
                        "is_clean": audit.get("is_clean", True),
                        "summary": audit.get("summary", ""),
                        "refactored_code": audit.get("refactored_code", "")
                    })

    return render_template(
        'clean_code.html',
        active_tab=active_tab,
        dirty_code=dirty_code,
        feedback=feedback,
        github_results=github_results,
        error=error
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
