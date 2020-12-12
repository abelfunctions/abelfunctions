import getopt
import sys
import warnings

from abelfunctions import __version__


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
    print(s)


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

    try:
        import pytest
    except ImportError as e:
        raise ImportError(str(e)+"\n\n"
            "Note: Running tests in parallel requires pytest-xdist.  Install with:\n\n"
            "\t$ sage -pip install pytest-xdist\n\n"
            "Note: If pip does not work because 'the ssl module in Python is not available' "
            "then you may need to install openssl into sage and rebuild python using\n\n"
            "\t$ sage -i openssl\n"
            "\t$ sage -f python3"
        )

    # determine list of search patterns for tests
    patterns = ' '.join(args)
    pytest_args = ['-k', patterns, '--ignore=examples']
    if processes > 1:
        pytest_args += ['-n', str(processes)]

    # highlight the runtimes for the 5 slowest tests
    pytest_args.append('--durations=5')

    if verbosity == 2:
        pytest_args.append('-v')
    errno = pytest.main(pytest_args)
    sys.exit(errno)

if __name__ == '__main__':
    # run tests and suppress warnings (particularly from PARI)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runtests(sys.argv[1:])
