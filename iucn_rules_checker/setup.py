from setuptools import setup, find_packages

setup(
    name="iucn_rules_checker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pytest>=7.0.0",
    ],
)
