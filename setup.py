from setuptools import setup

setup(
    name='fastatomography',
    version='0.1',
    packages=['fastatomography',
              'fastatomography.operators',
              'fastatomography.util',
              'fastatomography.reporting',
              'fastatomography.tomo'],
    package_dir={'fastatomography': 'fastatomography/',
                 'fastatomography.util': 'fastatomography/util',
                 'fastatomography.operators': 'fastatomography/operators',
                 'fastatomography.reporting': 'fastatomography/reporting',
                 'fastatomography.tomo': 'fastatomography/tomo'},
    url='',
    install_requires=["torch>=1.3"],
    license='GPLv3',
    author='Philipp Pelz',
    author_email='philipp.pelz@berkeley.edu',
    description='Tomography Reconstruction Toolbox',
    python_requires='>=3.6',
)
