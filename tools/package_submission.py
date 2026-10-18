"""Package source with an explicit allowlist; never include local credentials/caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
output = root.parent / 'TrueTone-submission-source.zip'
extensions = {'.py', '.html', '.js', '.json', '.md', '.txt', '.kt', '.kts', '.toml',
              '.xml', '.webp', '.properties', '.bat', '.jar', '.keep'}
excluded = {'.git', '.idea', '.gradle', '.kotlin', '.venv', '__pycache__', 'build',
            '.pytest_cache', 'runtime', 'local.properties', '.env'}
files = [root / 'SUBMISSION.md']
for folder in ('android', 'backend', 'behavior_ml', 'contracts', 'tests'):
    for path in (root / folder).rglob('*'):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in excluded or part.startswith('.env') for part in relative.parts):
            continue
        if path.suffix in extensions or path.name in {'gradlew', '.gitignore'}:
            files.append(path)
with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, 'TrueTone/' + path.relative_to(root).as_posix())
with ZipFile(output) as archive:
    assert archive.testzip() is None
print(f'{output}: {len(files)} files')
