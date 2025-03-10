from setuptools import setup, find_packages

setup(
    name="realbt",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "click",  # For CLI functionality
        "pyyaml",  # For configuration files
        "tqdm",    # For progress bars
    ],
    entry_points={
        "console_scripts": [
            "realbt=realbt.cli:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="REAListic BackTesting framework with accurate market friction modeling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/realbt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
)