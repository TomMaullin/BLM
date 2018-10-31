from setuptools import setup, find_packages

readme = "Empty for now"

reqs = [line.strip() for line in open('requirements.txt').readlines()]
requirements = list(filter(None, reqs))

setup(
    name="DistOLS-py",
    version="0.0.1",
    author="Tom Maullin",
    scripts=['bin/DistOLS'],
    author_email="tommaullin@gmail.com",
    description=(
        "Code for running distributed OLS."),
    packages=find_packages(),
    long_description=readme,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Programming Language :: Python :: 3",
    ],
    install_requires=requirements
)