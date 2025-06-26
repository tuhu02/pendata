from setuptools import setup, find_packages

with open("README_DEPLOYMENT.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="wdbc-breast-cancer-prediction",
    version="1.0.0",
    author="AI Assistant",
    author_email="assistant@example.com",
    description="A Streamlit web application for breast cancer prediction using WDBC dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wdbc-breast-cancer-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "wdbc-app=streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["models/*.pkl", "models/*.json", "*.csv"],
    },
    keywords="breast cancer, machine learning, streamlit, medical, prediction, wdbc",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/wdbc-breast-cancer-prediction/issues",
        "Source": "https://github.com/yourusername/wdbc-breast-cancer-prediction",
        "Documentation": "https://github.com/yourusername/wdbc-breast-cancer-prediction/blob/main/README_DEPLOYMENT.md",
    },
) 