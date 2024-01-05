from setuptools import find_packages, setup

setup(
    name='SumEvaluator',
    packages=find_packages(),
    version='0.1.0',
    description='A library designed for the analysis of various aspects of generated summaries.',
    author='Ala Falaki',
    license='MIT',
    install_requires=["torch", "nltk", "sentence_transformers", "pandas", "openai", "six"]
)