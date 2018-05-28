from setuptools import setup

setup(name='cortex',
      version='0.1',
      description='A library for wrapping your pytorch code',
      author='R Devon Hjelm',
      author_email='erroneus@gmail.com',
      packages=['lib'],
      install_requires=[
        'imageio', 'matplotlib', 'progressbar2', 'scipy', 'sklearn', 'torchvision', 'visdom', 'pyyaml'],
      entry_points={
        'console_scripts': [
            'cortex=lib:main']
      },
      zip_safe=False)