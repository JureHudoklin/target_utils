from setuptools import setup, find_packages

setup(
    name='Target Utils',
    version='0.1',
    package_dir={"": "src"},
    packages=find_packages('src'),
    install_requires=[
        'torch >= 2.1.0',
    ],
    author='Jure Hudoklin',
    description='A package for loading and processing data in "Swift" format',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
