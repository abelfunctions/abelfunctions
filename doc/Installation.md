# Installation

Abelfunctions requires [Sage (SageMath)](http://www.sagemath.org) 8.0 or later.
Sage makes it relatively easy to build and run the code.

> **Note:** Depending on your system one of the below installation methods
may not work. I'm not sure what is causing this. Any contributions to the
installation method (via `setup.py`) is greatly appreciated.

# Recommended Method

In a terminal, run the following commands

```bash
$ sage --pip install git+https://github.com/abelfunctions/abelfunctions
```

# Alternate method

1. Download Abelfunctions using [Git](https://git-scm.com) or by clicking
   on the *"Download Zip"* button on the right-hand side of the
   [repository page](https://github.com/abelfunctions/abelfunctions).
2. Enter the top-level directory, the one containing `setup.py` and run

   ```
   $ sage setup.py install
   ```

# Alternate Method

If the above does not work for whatever reason, try this instead:

1. Download Abelfunctions using [Git](https://git-scm.com) or by clicking on
   the *"Download Zip"* button on the right-hand side of the
   [repository page](https://github.com/abelfunctions/abelfunctions).

   a. In the latter case, make sure that the name of the directory containing
   the package is exactly `abelfunctions`, not something like
   `abelfunctions-master`; i.e., rename it if necessary.

2. Convert the entire project into a Sage SPKG:

   ```
   $ tar cjf abelfunctions.spkg /path/to/abelfunctions
   ```

   *Note: some day this step will be done for you or will be unnecessary.*

3. Install the SPKG into Sage:

   ```
   $ sage -p abelfunctions.spkg     # without running test suite
   $ sage -p -c abelfunctions.spkg  # with running test suite
   ```
