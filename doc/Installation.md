# Installation

Abelfunctions requires [Sage (SageMath)](http://www.sagemath.org) 9.2 or later.
Sage makes it relatively easy to build and run the code.

> **Note:** Depending on your system one of the below installation methods
may not work. I'm not sure what is causing this. Any contributions to the
installation method (via `setup.py`) is greatly appreciated.

# Recommended Method

In a terminal, run the following commands

```bash
sage --pip install --no-build-isolation git+https://github.com/abelfunctions/abelfunctions
```

# Alternate method

1. Download Abelfunctions using [Git](https://git-scm.com) or by clicking
   on the *"Download Zip"* button on the right-hand side of the
   [repository page](https://github.com/abelfunctions/abelfunctions).
2. Enter the top-level directory, the one containing `setup.py` and run

   ```
   sage setup.py install
   ```
