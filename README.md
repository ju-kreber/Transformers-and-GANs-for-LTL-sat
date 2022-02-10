Transformers and GANs for LTL-sat
=================================

This is the implementation of the Master thesis of Jens Ulrich Kreber titled "Generating and Solving Temporal Logic Problems with Adversarial Transformers".
It contains Transformer-based architectures for solving linear-time temporal logic satisfiability and for the generation of new challenging instances with a new type of Wasserstein GAN.
Use of this code is detailed in the reproducibility section of the thesis.


## Installation
The code is shipped as a Python package that can be installed by executing

    pip install -e .

in the `impl` directory (where `setup.py` is located). Python version 3.6 or higher is required.
Additional dependencies such as `tensorflow` will be installed automatically.
To generate datasets or solve LTL formulas immediately after generation, the LTL satisfiability checking tool `aalta` is required as binary.
It can be obtained from [bitbucket](https://bitbucket.org/jl86/aalta) (earliest commit in that repository).
After compiling, ensure that the binary `aalta` resides under the `bin` folder.

