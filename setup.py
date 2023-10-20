from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='core',
    version='0.1.0',
    description='Core repo for the Gesture Control project',
    author='Patrick Strasser',
    author_email='patrick.strasser@stud-mail.uni-wuerzburg.de',
    packages=find_packages(),
    install_requires=requirements,
)
