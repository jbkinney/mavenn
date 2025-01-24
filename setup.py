from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='mavenn',
      version='1.1.1',
      description='MAVE-NN: learning genotype-phenotype maps from multiplex assays of variant effect',
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      keywords='genotype-phenotype maps, multiplex assays, variant effect, deep mutational scanning, massively parallel reporter assays',
      url='https://mavenn.readthedocs.io',
      author='Ammar Tareen and Justin B. Kinney',
      author_email='jkinney@cshl.edu',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[      # tested on 2025.01.23
        'numpy',              # tested with version 1.26.4
        'matplotlib',         # tested with version 3.10.0
        'pandas',             # tested with version 2.2.3
        'tensorflow',         # tested with version 2.17.0
        'scikit-learn',       # tested with version 1.6.1
        'scipy',              # tested with version 1.15.1
        'seaborn',            # tested with version 0.13.2
        'logomaker',          # tested with version 0.8.2
      ],
      python_requires='>=3.8',# tested with version 3.12.8
      zip_safe=False)