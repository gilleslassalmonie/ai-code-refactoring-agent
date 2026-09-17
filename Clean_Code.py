"""Flask application that reviews source code with LangChain and an OpenAI model.

The HTML template posts one of three workflows to the root route: an educational
review, a quick refactor, or a public GitHub repository audit.  This module keeps
the web layer separate from LLM calls and source-file collection so each part can
be changed or tested independently.

Required environment variable: OPENAI_API_KEY
Optional environment variable: CLEAN_CODE_MODEL (defaults to gpt-4.1-mini)
"""

# Standard-library modules handle files, ZIP archives, URLs, and pattern matching.
import os
import io
import re
import zipfile

# requests is only used for public GitHub API and raw-file calls.
import requests
from urllib.parse import quote

# Flask request field names deliberately match clean_code.html.
from flask import Flask, render_template, request

# LangChain provides a provider-neutral interface.  This app uses the OpenAI
# integration, but the Flask routes below do not need to change if a different
# LangChain chat model is selected later.
try:
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except ImportError:
    JsonOutputParser = StrOutputParser = ChatPromptTemplate = ChatOpenAI = None

# Flask automatically searches for clean_code.html in a sibling templates folder.
app = Flask(__name__)
# Reject unexpectedly large uploads before loading them into memory.
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
    # splitext is safer than checking suffixes manually because it normalizes
    # multi-character extensions such as .cpp and ignores the directory portion.
    extension = os.path.splitext(filename.lower())[1]
    return LANGUAGE_BY_EXTENSION.get(extension, 'Unknown')


def is_supported_source_file(filename: str) -> bool:
    """Keep unsupported files out of ZIP extraction and GitHub processing."""
    return language_for_filename(filename) != 'Unknown'


def detect_language_from_code(code: str) -> str:
    """Best-effort language detection for code pasted without a filename."""
    normalized = code.strip()
    # These checks are intentionally conservative.  They only guide the review
    # prompt; they do not parse, execute, or validate untrusted user code.
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
    """LangChain-backed reviewer for the three modes exposed by the HTML page."""

    def __init__(self, model_name: str | None = None):
        # Keep the model configurable without exposing another form field.  This
        # lets deployment set CLEAN_CODE_MODEL while local users use the default.
        self.model_name = model_name or os.getenv("CLEAN_CODE_MODEL", "gpt-4.1-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")

    def _configuration_error(self) -> str | None:
        """Return an actionable message rather than failing during a POST request."""
        if ChatOpenAI is None:
            return (
                "Error: LangChain is not installed. Run "
                "`pip install langchain-core langchain-openai`."
            )
        if not self.api_key:
            return "Error: OPENAI_API_KEY is missing from environment variables."
        return None

    def _chat_model(self, temperature: float) -> ChatOpenAI:
        """Create a fresh LangChain chat model with the mode's creativity level."""
        return ChatOpenAI(
            model=self.model_name,
            temperature=temperature,
            api_key=self.api_key,
        )

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
            "- 'refactored_code': string with clean code in the file's original language if not clean, else original"
        )
        configuration_error = self._configuration_error()
        if configuration_error:
            return {"is_clean": True, "summary": configuration_error, "refactored_code": code}

        try:
            # JsonOutputParser validates that the model returns a JSON object and
            # converts it to a Python dictionary for the existing HTML template.
            parser = JsonOutputParser()
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_instruction),
                ("human", "File: {filename}\n\nCode:\n{code}\n\n{format_instructions}"),
            ])
            chain = prompt | self._chat_model(temperature=0.1) | parser
            audit = chain.invoke({
                "filename": filename,
                "code": code,
                "format_instructions": parser.get_format_instructions(),
            })
            return {
                "is_clean": bool(audit.get("is_clean", True)),
                "summary": str(audit.get("summary", "")),
                "refactored_code": str(audit.get("refactored_code", code)),
            }
        except Exception as e:
            return {"is_clean": True, "summary": f"Error auditing file: {str(e)}", "refactored_code": code}

    def _generate(self, prompt: str, system_instruction: str, temperature: float) -> str:
        configuration_error = self._configuration_error()
        if configuration_error:
            return configuration_error
        try:
            # A prompt template keeps instructions separate from user-provided
            # code.  StrOutputParser extracts the text from LangChain's AIMessage.
            chain = (
                ChatPromptTemplate.from_messages([
                    ("system", system_instruction),
                    ("human", "Code to review:\n{code}"),
                ])
                | self._chat_model(temperature)
                | StrOutputParser()
            )
            return chain.invoke({"code": prompt})
        except Exception as e:
            return f"Error during analysis: {str(e)}"


def extract_source_code_from_upload(file_storage) -> str:
    """Extract supported source files from an upload or a ZIP archive."""
    # Werkzeug provides the upload as a stream.  Lowercasing makes extension
    # handling consistent for files such as PROGRAM.JS and program.js.
    filename = file_storage.filename.lower()
    
    if is_supported_source_file(filename):
        # errors='ignore' lets the reviewer handle imperfectly encoded source
        # files instead of failing the entire HTTP request.
        return file_storage.read().decode('utf-8', errors='ignore')
        
    elif filename.endswith('.zip'):
        extracted_code = []
        with zipfile.ZipFile(io.BytesIO(file_storage.read())) as z:
            for zip_info in z.infolist():
                # Ignore directory entries and macOS metadata.  Only listed
                # source extensions are read, preventing binary files entering
                # the LLM prompt.
                if (not zip_info.is_dir() and not zip_info.filename.startswith('__MACOSX')
                        and is_supported_source_file(zip_info.filename)):
                    with z.open(zip_info) as f:
                        code_content = f.read().decode('utf-8', errors='ignore')
                        language = language_for_filename(zip_info.filename)
                        extracted_code.append(
                            # A per-file marker helps the model distinguish files
                            # when a ZIP contains a multi-language project.
                            f"// --- File: {zip_info.filename} ({language}) ---\n{code_content}\n"
                        )
        return "\n".join(extracted_code) if extracted_code else "No supported source files found in ZIP archive."
    
    return ""


def fetch_github_source_files(repo_url: str) -> dict:
    """Fetch supported source files from a public GitHub repository."""
    # The route accepts a normal GitHub URL, so extract only owner/repository
    # segments before constructing GitHub API URLs.
    match = re.search(r"github\.com/([^/]+)/([^/.]+)", repo_url)
    if not match:
        return {"error": "Invalid GitHub repository URL format."}

    owner, repo = match.group(1), match.group(2)
    # First read repository metadata rather than assuming main/master.  This
    # supports repositories whose default branch has a custom name.
    repo_response = requests.get(f"https://api.github.com/repos/{owner}/{repo}", timeout=15)
    if repo_response.status_code != 200:
        return {"error": f"Failed to fetch repository. Status code: {repo_response.status_code}"}

    default_branch = repo_response.json().get('default_branch', 'main')
    # quote prevents a branch name containing URL-reserved characters from
    # changing the API request path.
    api_url = (
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/"
        f"{quote(default_branch, safe='')}?recursive=1"
    )
    res = requests.get(api_url, timeout=15)
    if res.status_code != 200:
        return {"error": f"Failed to fetch repository. Status code: {res.status_code}"}

    # A recursive tree returns file metadata, not file bodies.  Download only
    # supported blob entries in the loop below.
    tree = res.json().get('tree', [])
    source_files = {}

    for item in tree:
        if item['type'] == 'blob' and is_supported_source_file(item['path']):
            # Keep slashes in the file path so nested directories remain valid
            # while escaping characters that could break the raw GitHub URL.
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
    """Render the page on GET and dispatch the selected HTML tab on POST."""
    # `active_tab` is a hidden input in each form.  Passing it back to the
    # template ensures the user returns to the same tab after submission.
    active_tab = request.form.get('active_tab', 'edu')
    dirty_code = ""
    feedback = None
    github_results = None
    error = None

    if request.method == 'POST':
        # Create one reviewer per request.  The model client has no request
        # state, which avoids accidentally sharing user code between requests.
        reviewer = CleanCodeReviewer()

        # MODE 1: Educational
        if active_tab == 'edu':
            if 'file_upload' in request.files and request.files['file_upload'].filename != '':
                # Field name is kept in sync with the educational upload input
                # in clean_code.html.
                uploaded_file = request.files['file_upload']
                dirty_code = extract_source_code_from_upload(uploaded_file)
                language = (
                    'multiple source languages' if uploaded_file.filename.lower().endswith('.zip')
                    else language_for_filename(uploaded_file.filename)
                )
            else:
                # Pasted code has no extension, so use lightweight detection to
                # provide the LLM with the most appropriate language context.
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
                # Keep the result keys stable because the HTML uses
                # item.file, item.is_clean, item.summary, and item.refactored_code.
                for filepath, code_content in repo_data["files"].items():
                    audit = reviewer.review_github_file(filepath, code_content)
                    github_results.append({
                        "file": filepath,
                        "is_clean": audit.get("is_clean", True),
                        "summary": audit.get("summary", ""),
                        "refactored_code": audit.get("refactored_code", "")
                    })

    # The template receives the same context keys for GET and POST.  Empty
    # values are harmless and let Jinja conditionally hide result sections.
    return render_template(
        'clean_code.html',
        active_tab=active_tab,
        dirty_code=dirty_code,
        feedback=feedback,
        github_results=github_results,
        error=error
    )


if __name__ == '__main__':
    # Debug mode is suitable for local development only.  Use a production WSGI
    # server and disable debug mode when deploying the application publicly.
    app.run(debug=True, port=5000)
