from setuptools import setup

packages = [
    'cortex', 'cortex._lib', 'cortex.built_ins', 'cortex._lib.data',
    'cortex.built_ins.datasets', 'cortex.built_ins.models',
    'cortex.built_ins.networks', 'cortex.built_ins.transforms']

install_requirements = [
    'imageio', 'matplotlib', 'progressbar2', 'scipy',
    'sklearn', 'visdom', 'pyyaml', 'pathlib',
    'sphinxcontrib-napoleon', 'nibabel'
]

setup(name='cortex',
      version='0.11',
      description='A library for wrapping your pytorch code',
      author='R Devon Hjelm',
      author_email='erroneus@gmail.com',
      packages=packages,
      install_requires=[
          'imageio', 'matplotlib', 'progressbar2', 'scipy', 'sklearn',
          'torchvision', 'visdom', 'pyyaml'],
      entry_points={
          'console_scripts': [
              'cortex=cortex.main:run']
      },
      zip_safe=False)
