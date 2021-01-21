# coding=utf-8
# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

setup(
    name='reranker',
    version='0.0.1',
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        'torch>=1.6.0',
        'transformers>=4.0.0',
        'datasets>=1.1.3',
    ],
    url='',
    license='CC-BY-NC 4.0',
    author='Luyu Gao',
    author_email='luyug@cs.cmu.edu',
    description=''
)
