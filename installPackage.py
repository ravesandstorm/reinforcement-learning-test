import subprocess
import sys

def install_package(package):
    subprocess.run([sys.executable, "-m", "pip", "install", package])

# Example usage
install_package("torch")