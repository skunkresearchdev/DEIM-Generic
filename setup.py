"""
Setup script for DEIM module
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)

setup(
    name='deim',
    version='1.0.0',
    author='DEIM Contributors',
    description='DEIM - DETR with Improved Matching for object detection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/deim',
    packages=find_packages(),
    package_data={
        'deim': [
            '_configs/*.yml',
            '_configs/_base/*.yml',
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='object detection, deep learning, computer vision, DEIM, DETR',
    entry_points={
        'console_scripts': [
            'deim=deim.api:main',
        ],
    },
)