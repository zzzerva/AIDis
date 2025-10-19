"""
AIDis: Disleksi Analiz Sistemi
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aidis",
    version="1.0.0",
    author="AIDis Team",
    author_email="team@aidis.com",
    description="Disleksi analizi için yapay zeka sistemi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AIDisMihri",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
            "mypy",
            "pytest",
            "pytest-cov",
            "pre-commit",
        ],
        "web": [
            "Django>=4.2",
            "Pillow",
            "reportlab",
        ],
        "ml": [
            "tensorflow>=2.12",
            "opencv-python",
            "scikit-learn",
            "matplotlib",
            "numpy",
        ],
    },
    entry_points={
        "console_scripts": [
            "aidis-train=AIDis_model.src.train:main",
            "aidis-evaluate=AIDis_model.src.evaluate:main",
            "aidis-prepare=AIDis_model.src.prepare_data:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
