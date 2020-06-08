import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gluon-utils",
    version="0.0.1",
    author="Caleb Vatral",
    author_email="caleb.m.vatral@vanderbilt.edu",
    description="Collection of utilities to complement or extend the functionality of Gluon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kbvatral/gluon-utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'numpy',
          'matplotlib',
          'tqdm',
      ],
    python_requires='>=3.5',
)