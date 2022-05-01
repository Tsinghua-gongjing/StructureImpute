"""
# Author: Jing Gong, Kui XU
# File Name: setup.py
# Description:
"""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='StructureImpute',
      version='0.1.1',
      description='StructureImpute',
      packages=find_packages(),

      author='Jing Gong',
      author_email='gong15@tsinghua.org',
      url='https://github.com/Tsinghua-gongjing/StructureImpute',
      install_requires=requirements,
      python_requires='>3.6.0',

      classifiers=[
          'Development Status :: 1 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: CentOS :: Linux',
     ],
     )