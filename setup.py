from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='dfcs_vipa',
    version='1.0.0',
    description="Direct frequency comb spectroscopy with a VIPA spectrograph",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/gkowzan/dfcs_vipa',
    author='Grzegorz Kowzan',
    author_email='grzegorz@kowzan.eu',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Physics'],
    python_requires='>=3.5',
    install_requires=['numpy', 'scipy', 'xarray', 'h5py', 'lxml'],
    extras_require={'doc': ['sphinx', 'sphinx_rtd_theme', 'numpydoc'],
                    'plot': ['matplotlib']},
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "labview_grid = dfcs_vipa.scripts.labview_grid:main",
            "combwatch = dfcs_vipa.scripts.combwatch:main",
            "hdf5_convert = dfcs_vipa.scripts.hdf5_convert:main"]}
)
