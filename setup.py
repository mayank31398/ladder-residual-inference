from setuptools import find_packages, setup

setup(
    name="gpt-fast",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
    ],
    description="A simple, fast, pure PyTorch Llama inference engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch-labs/gpt-fast",
)
