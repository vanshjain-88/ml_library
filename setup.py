from setuptools import setup, find_packages

setup(
    name='ml_from_scratch',
    version='0.1.0',
    description='A simple machine learning library implemented from scratch.',
    author='Vansh Jain',
    author_email='vanshjain2411@gmail.com',
    packages=find_packages(include=['Modules', 'Modules.*']),
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    python_requires='>=3.6',
)
