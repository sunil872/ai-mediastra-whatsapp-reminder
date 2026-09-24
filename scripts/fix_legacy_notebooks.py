import json
from pathlib import Path

paths = ['RefillCare_Phases_1_to_3_Walkthrough.ipynb', 'notebooks/RefillCare_Phases_1_to_3_Walkthrough.ipynb']

for p_str in paths:
    p = Path(p_str)
    if not p.exists():
        print(f"Skipping {p} (does not exist)")
        continue
    with open(p, 'r', encoding='utf-8', errors='replace') as f:
        nb = json.loads(f.read(), strict=False)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell['source'] = [
                "import sys\n",
                "import os\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import json\n",
                "try:\n",
                "    from IPython.display import display\n",
                "except ImportError:\n",
                "    display = print\n",
                "try:\n",
                "    import matplotlib.pyplot as plt\n",
                "except ImportError:\n",
                "    plt = None\n",
                "\n",
                "current_dir = Path(__file__).resolve().parent if '__file__' in locals() else Path.cwd()\n",
                "project_root = current_dir.resolve()\n",
                "while project_root.parent != project_root and not (project_root / 'refillcare' / '__init__.py').exists():\n",
                "    project_root = project_root.parent\n",
                "if (project_root / 'refillcare' / '__init__.py').exists() and str(project_root) not in sys.path:\n",
                "    sys.path.insert(0, str(project_root))\n",
                "\n",
                "def find_file(relative_path: str) -> Path:\n",
                "    candidates = [\n",
                "        project_root / relative_path,\n",
                "        Path.cwd() / relative_path,\n",
                "        Path('..') / relative_path,\n",
                "        Path('../..') / relative_path,\n",
                "        Path(r'C:\\Users\\sunil\\ai-mediastra-whatsapp-reminder\\ai-mediastra-whatsapp-reminder') / relative_path,\n",
                "    ]\n",
                "    for c in candidates:\n",
                "        if c.exists():\n",
                "            return c.resolve()\n",
                "    return candidates[0]\n",
                "print(f'[PASS] Project Root resolved: {project_root}')\n"
            ]
            break
            
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f'Fixed {p}')
