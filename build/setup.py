"""
Setup script para instalação e distribuição do MedAI
"""

from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).parent.parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

requirements_path = Path(__file__).parent.parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')

setup(
    name="medai-radiologia",
    version="1.0.0",
    author="Dr. Guilherme Capel",
    author_email="drguilhermecapel@gmail.com",
    description="Sistema de análise de imagens radiológicas médicas por IA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drguilhermecapel/radiologyai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "medai=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
)
