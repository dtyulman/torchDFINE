from setuptools import setup, find_packages
import glob, os

top_level_modules = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob("*.py") if p not in ("setup.py",)]

setup(
    name="torchdfine",
    version="0.1",
    packages=find_packages(),
    py_modules=top_level_modules,
)
