from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='laab_python',
    version="v2025",
    description="Linear Algebra Awareness Benchmark for Python frameworks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/HPAC/LAAB-Python',
    author='Aravind Sankaran',
    author_email='aravind.sankaran@rwth-aachen.de',
    packages= find_packages(), # finds packages inside current directory
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">3.6",
    install_requires=open("requirements.txt").read().splitlines(),

)
