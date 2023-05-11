import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requires = fh.read()

setuptools.setup(
    description="A Hair Cell Analysis Toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.9',
    entry_points={'console_scripts': [TODO]},
    install_requires=requires
)
