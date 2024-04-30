from setuptools import setup

setup(
    author="Pierre-Antoine Bannier",
    author_email="pierre-antoine.bannier@owkin.com",
    name='fgfr3mut',
    version='1.0',
    python_requires=">=3.8,<3.10",
    install_requires=[
        "torch==2.3.0",
        "numpy==1.23.5",
        "pandas==1.1.3",
        "xlrd==1.1.0",
        "tqdm==4.66.2",
        "loguru==0.7.2",
        "scikit-learn==1.3.0",
        "huggingface-hub==0.22.2"
    ],
    packages=["fgfr3mut"],
    package_dir={"fgfr3mut": "fgfr3mut"},
)
