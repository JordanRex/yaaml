import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yaaml",
    version="0.0.2",
    author="Varun Rajan",
    author_email="varunrajan7@gmail.com",
    description="Yet-Another-Auto-ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JordanRex/yaaml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows :: Windows 10",
    ],
)