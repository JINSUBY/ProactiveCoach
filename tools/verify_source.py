"""Check archived source hashes and syntax without importing research modules."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'docs/source-manifest.json').read_text(encoding='utf-8'))
    errors = []
    for entry in manifest:
        path = root / entry['path']
        if not path.is_file():
            errors.append(f"Missing: {entry['path']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            errors.append(f"Changed from supplied archive: {entry['path']}")
    python_files = sorted(root.rglob('*.py'))
    json_files = sorted(root.rglob('*.json'))
    # Scope to release code/documentation, not local environments or datasets.
    allowed = {'evaluation', 'human_eval', 'proactivecoach', 'scripts', 'tools', 'docs'}
    python_files = [p for p in python_files if len(p.relative_to(root).parts) == 1
                    or p.relative_to(root).parts[0] in allowed]
    json_files = [p for p in json_files if len(p.relative_to(root).parts) == 1
                  or p.relative_to(root).parts[0] in allowed]
    for path in python_files:
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            compile(tree, str(path), 'exec')
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f'{path.relative_to(root)}: {exc}')
    for path in json_files:
        try:
            json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, UnicodeError) as exc:
            errors.append(f'{path.relative_to(root)}: {exc}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(manifest)} original hashes; {len(python_files)} Python syntax checks; '
          f'{len(json_files)} JSON parses. No research modules executed.')


if __name__ == '__main__':
    main()
