# Installation

Abelfunctions requires [Sage (SageMath)](http://www.sagemath.org) 9.2 or later.
Sage makes it relatively easy to build and run the code.

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

   ```bash
   sage --pip install --no-build-isolation --editable .
   ```

   With the optional flag `--editable`, any changes to Python modules in
   the directory will take immediate effect after restarting the Sage
   session. If you make changes to Cython sources, repeat the above
   command so that the modules are recompiled.
