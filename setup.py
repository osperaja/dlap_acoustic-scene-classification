from setuptools import setup, find_packages

VERSION = '1.0.0'
AUTHOR = 'Jakob Kienegger and Navin Raj Prabhu'
AUTHOR_EMAIL = 'jakob.kienegger@uni-hamburg.de'
DESCRIPTION = 'DCASE Challenge Framework'

def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()

# Setting up
setup(
    name="dcase",
	author=AUTHOR,
	author_email=AUTHOR_EMAIL,
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=read_requirements('requirements.txt')
)
