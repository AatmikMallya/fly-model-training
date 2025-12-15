import os
import subprocess
import re
import json

def find_imports(directory='.'):
    """Scan Python files and Jupyter Notebooks for import statements and collect package names."""
    packages = set()
    import_pattern = re.compile(r'^\s*(?:import|from)\s+([\w.]+)')

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        match = import_pattern.match(line)
                        if match:
                            package = match.group(1).split('.')[0]
                            packages.add(package)
            elif file.endswith('.ipynb'):
                with open(os.path.join(root, file), 'r') as f:
                    notebook = json.load(f)
                    for cell in notebook.get('cells', []):
                        if cell['cell_type'] == 'code':
                            for line in cell['source']:
                                match = import_pattern.match(line)
                                if match:
                                    package = match.group(1).split('.')[0]
                                    packages.add(package)
    return packages

def install_packages(packages):
    """Install the found packages using pip."""
    for package in packages:
        subprocess.call(['pip', 'install', package])

if __name__ == '__main__':
    print("Scanning for packages...")
    packages = find_imports()
    print(f"Found packages: {packages}")
    install_packages(packages)
    print("Installation complete.")
