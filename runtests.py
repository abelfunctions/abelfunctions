import abelfunctions
import getopt
import sys
import unittest
import warnings

# get abelfunctions version number as '__version__'
exec(open('abelfunctions/version.py')).read()


def usage():
    s = """
------------------------------------------------------------
 Abelfunctions Version %s Test Suite
------------------------------------------------------------

Usage:

    $ sage -python runtests [-hv] [module name]

Optional arguments:

    [module_name] -- if provided, only run tests matching name
    -h            -- print this help message
    -v            -- run tests with higher verbosity level
    -p <arg>      -- run tests in parallel using <arg> number of processes (requires pytest-xdist)
    
"""%__version__
    print s


def runtests(argv):
    # obtain command line arguments
    try:
        opts, args = getopt.getopt(argv, 'hvp:m')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    # parse command line arguments. set default values
    verbosity = 1
    processes = 1
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        if opt == '-v':
            verbosity = 2
        if opt == '-p':
            processes = int(arg)

    if processes > 1:
        try:
            import pytest
        except ImportError:
            raise ImportError("Runinng tests in parallel requires pytest-xdist.  Install with:\n\n"
                       "\t$ sage -pip install pytest-xdist \n")

        # determine list of search patterns for tests
        patterns = ''
        for arg in args:
            patterns += str(arg) + ' '
        patterns = patterns[:-1]

        pytest_args = ['-k', patterns, '--ignore=examples', '-n', processes]

        # highlight the runtimes for the 5 slowest tests
        pytest_args.append('--durations=5')

        if verbosity == 2:
            pytest_args.append('-v')
        errno = pytest.main(pytest_args)

    else:
        # determine list of search patterns for tests
        patterns = []
        for arg in args:
            # if the string 'test' doesn't appear in the pattern argument then
            # prepend to string
            pattern = arg
            if not 'test' in pattern:
                pattern = 'test*' + pattern + '*'
            if not '.py' in pattern:
                pattern = pattern + '.py'
            patterns.append(pattern)
        if not patterns:
            patterns = ['test*.py']

        # run tests for each requested pattern
        start_dir = 'abelfunctions'
        for pattern in patterns:
            loader = unittest.TestLoader()
            suite = loader.discover(start_dir, pattern=pattern)
            runner = unittest.TextTestRunner(verbosity=verbosity)
            result = runner.run(suite)
            errno = not result.wasSuccessful()

    sys.exit(errno)

if __name__ == '__main__':    
    # run tests and suppress warnings (particularly from PARI)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runtests(sys.argv[1:])
