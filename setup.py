from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='dfcs_vipa',
    version='1.0.0',
    description="Frequency comb spectroscopy with VIPA",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://bitbucket.org/yoreh/dfcs_vipa',
    author='Grzegorz Kowzan',
    author_email='grzegorz@kowzan.eu',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Physics'],
    python_requires='>=3.5',
    install_requires=['numpy', 'scipy', 'xarray', 'h5py', 'lxml'],
    extras_require={'doc': ['sphinx', 'sphinx_rtd_theme', 'numpydoc']},
    packages=find_packages()
)
