"""Run experiments.ipynb and save its outputs."""
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('IPYTHONDIR', str(ROOT / '.cache' / 'ipython'))
os.environ.setdefault('JUPYTER_RUNTIME_DIR', str(ROOT / '.cache' / 'jupyter'))
for name in ('MPLCONFIGDIR', 'IPYTHONDIR', 'JUPYTER_RUNTIME_DIR'):
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)

path = ROOT / 'experiments.ipynb'
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name='python3')
# The system Jupyter kernel may use Python from a different environment.
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(notebook, km=manager, timeout=1800, resources={'metadata': {'path': str(ROOT)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
print(f'Saved {path.name}')
