from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name='protected-classification',
  packages=['protected_classification'],
  package_dir={'protected_classification': 'src'},
  version='0.1.0',
  license='MIT',
  description='Protected Classification package',
  long_description=long_description,
  long_description_content_type="text/markdown",
  author='Ivan Petej',
  author_email='ivan.petej@gmail.com',
  url='https://github.com/ip200/protected-classification',
  download_url='https://github.com/ip200/protected-classification/archive/refs/tags/v0_1_0.tar.gz',
  keywords=['Probabilistic classification', 'calibration'],
  install_requires=[
          'numpy',
          'scikit-learn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    "License :: OSI Approved :: MIT License",
    'Programming Language :: Python :: 3',
  ],
)
