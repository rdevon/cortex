| System | 2.7 | 3.5 |
| --- | --- | --- |
| Linux CPU | [![Build Status](https://travis-ci.org/mila-udem/metaopt.svg?branch=master)](https://travis-ci.org/mila-udem/metaopt) | [![Build Status](https://travis-ci.org/mila-udem/metaopt.svg?branch=master)](https://travis-ci.org/mila-udem/metaopt) |
| Linux GPU | [![Build Status](https://travis-ci.org/mila-udem/metaopt.svg?branch=master)](https://travis-ci.org/mila-udem/metaopt) | [![Build Status](https://travis-ci.org/mila-udem/metaopt.svg?branch=master)](https://travis-ci.org/mila-udem/metaopt) |
| macOS CPU | [![Build Status](https://travis-ci.org/mila-udem/metaopt.svg?branch=master)](https://travis-ci.org/mila-udem/metaopt) | [![Build Status](https://travis-ci.org/mila-udem/metaopt.svg?branch=master)](https://travis-ci.org/mila-udem/metaopt) |

# Asynchronous Hyperparameter Optimization on Distributed N-Node CPU/GPUs

# Requirements

- cmake 3.8.0 (minimum)
- gcc-5.0 (minimum)
- boost 1.55.1 (minimum)

# Installation

```bash
mkdir build
cd build
cmake..
make -j8
(sudo) make install
```

This will install `metaoptd` and `metaopt` binary in your computer. You will then need to install the `metaopt` Python package by

`python setup.py install`
