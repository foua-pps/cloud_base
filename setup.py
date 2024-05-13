#!/usr/bin/env python3
from pathlib import Path
import re
import setuptools

here = Path(__file__).parent.absolute()
required = [
    r for r in (here / 'requirements.txt').read_text().splitlines()
    if '=' in r or "git" in r
]
long_description = """
    Package for a cloud base product using VIIRS
"""

setuptools.setup(
    name='cbase',
    description='Package for cloud base algorithm',
    author='Inderpreet Kaur',
    author_email='inderpreet.kaur@smhi.se',
    url='http://nwcsaf.org',
    long_description=long_description,
    license='GPL',
    packages=setuptools.find_packages(),
    python_requires='>=3.9, <4',
    install_requires=required,
    include_package_data=True,
)
