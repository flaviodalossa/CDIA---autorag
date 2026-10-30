"""Build a source-only GitHub delivery, excluding all user data and credentials."""
from pathlib import Path
import sys
import zipfile

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
TOP_FILES = ['README.md', 'APLICACAO_WEB.md', '.gitignore', '.github/workflows/ci.yml', '.github/workflows/cloudflare-checks.yml']
APP_FILES = {'.gitignore', '.dockerignore', '.env.example', 'Dockerfile', 'compose.yaml', 'package.json', 'wrangler.jsonc'}
EXTENSIONS = {'.py', '.bat', '.ps1', '.md', '.txt', '.js', '.mjs', '.html', '.css'}


def source_files():
    for name in TOP_FILES:
        path = ROOT/name
        if path.is_file():
            yield path
    for path in APP.iterdir():
        if path.is_file() and (path.name in APP_FILES or path.suffix in {'.py','.bat','.md'} or path.name.startswith('requirements') and path.suffix=='.txt'):
            yield path
    cloudflare = APP/'cloudflare'
    if cloudflare.is_dir():
        config_names = {'package.json', 'pnpm-lock.yaml', 'pnpm-workspace.yaml', 'tsconfig.json', 'wrangler.jsonc', 'Dockerfile', '.gitignore', 'models.lock.json'}
        for path in cloudflare.iterdir():
            if path.is_file() and (path.name in config_names or path.suffix in {'.py', '.md', '.ts', '.mjs'}):
                yield path
        for folder in ['src', 'scripts', 'tests', 'licenses']:
            for path in (cloudflare/folder).rglob('*'):
                if path.is_file() and path.suffix in {'.py', '.ts', '.js', '.mjs', '.md', '.txt'} and '__pycache__' not in path.parts:
                    yield path
    for folder in ['autorag','web','scripts','tests']:
        for path in (APP/folder).rglob('*'):
            if path.is_file() and path.suffix in EXTENSIONS and not any(part in {'__pycache__','.venv','node_modules','.git'} for part in path.relative_to(APP).parts):
                yield path


def package(destination):
    destination=Path(destination).resolve()
    if destination.suffix.lower()!='.zip':
        raise ValueError('O destino deve ser um arquivo ZIP.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    files=sorted(set(source_files()))
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path,path.relative_to(ROOT).as_posix())
    print(f'{destination}: {len(files)} arquivos de código; sem dados, índices ou credenciais.')


if __name__=='__main__':
    package(sys.argv[1] if len(sys.argv)>1 else ROOT/'autorag-web-v0.4.zip')
