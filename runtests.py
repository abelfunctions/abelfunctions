import abelfunctions
import getopt
import sys
import unittest

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

"""%__version__
    print s


def runtests(argv):
    # obtain command line arguments
    try:
        opts, args = getopt.getopt(argv, 'hvm')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    # parse command line arguments. set default values
    verbosity = 1
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        if opt == '-v':
            verbosity = 2

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
        result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
        errno = not result.wasSuccessful()
    sys.exit(errno)

if __name__ == '__main__':
    runtests(sys.argv[1:])
