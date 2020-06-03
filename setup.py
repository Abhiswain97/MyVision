from setuptools import setup, Extension
from setuptools import find_packages

import MyVision

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


if __name__ == "__main__":
    setup(
        name="MyVision",
        version=MyVision.__version__,
        description="MyTorch: Collection of things I love about PyTorch",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Abhishek Swain",
        author_email="abhiswain.it.2016@gmail.com",
        url="https://github.com/Abhiswain97/MyVision",
        license="MIT License",
        packages=find_packages(),
        include_package_data=True,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        platforms=["linux", "unix"],
        python_requires=">3.5.2",
        install_requires=["scikit-learn>=0.22.1"],
    )