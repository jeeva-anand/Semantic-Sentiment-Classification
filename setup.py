
from setuptools import setup, find_packages


def get_requirements(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('-e .')]


setup(
    name="semantic-sentiment-analysis",
    version="0.0.1",
    author="Anand Raman",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
