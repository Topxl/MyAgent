"""
Setup script for MyAgent package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="myagent",
    version="0.1.0",
    author="MyAgent Team",
    author_email="info@myagent.example.com",
    description="Advanced open-source voice analysis & transcription framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/myagent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
        "librosa>=0.8.1",
        "numpy>=1.20.0",
        "soundfile>=0.10.3",
        "faster-whisper>=0.5.0",
        "speechbrain>=0.5.12",
        "fastapi>=0.75.0",
        "uvicorn>=0.17.6",
        "python-multipart>=0.0.5",
        "noisereduce>=2.0.0",
        "pydantic>=1.9.0",
    ],
    extras_require={
        "demucs": ["demucs>=4.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "black>=22.1.0",
            "flake8>=4.0.0",
            "mypy>=0.931",
        ],
    },
    entry_points={
        "console_scripts": [
            "myagent-server=myagent.api.server:run",
        ],
    },
)
