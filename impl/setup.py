from setuptools import setup, find_packages

setup(
    name='dlsgs',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'tensorflow>=2.3.0',
        'pycosat',
        'sympy>=1.7.1',
        'multiexit',
    ],
    python_requires='>=3.6'
)
