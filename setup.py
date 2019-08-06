#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import os

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

def get_all_data_files(directory='./real_robots/data/'):
    ret = []
    for dirpath, subdirs, files in os.walk(directory):
        for f in files:
            ret.append(os.path.join(dirpath, f))
    return ret


requirements = ['Click>=6.0', 'gym', 'pybullet', 'numpy', 'tqdm']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="S.P. Mohanty",
    author_email='mohanty@aicrowd.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Robots that learn to interact with the environment autonomously",
    entry_points={
        'console_scripts': [
            'real_robots=real_robots.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='real_robots',
    name='real_robots',
    packages=find_packages(include=['real_robots']),
    data_files=[('data', get_all_data_files())],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/spMohanty/real_robots',
    version='0.1.5',
    zip_safe=False,
)
