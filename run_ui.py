"""Launch the Streamlit UI: ``python run_ui.py [streamlit args]``."""
import subprocess
import sys
from pathlib import Path


def main() -> int:
    app = Path(__file__).resolve().parent / 'ui' / 'app.py'
    cmd = [sys.executable, '-m', 'streamlit', 'run', str(app), '--browser.gatherUsageStats=false', *sys.argv[1:]]
    return subprocess.call(cmd, cwd=str(Path(__file__).resolve().parent))


if __name__ == '__main__':
    sys.exit(main())
