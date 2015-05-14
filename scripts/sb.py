from optparse import OptionParser
import sys

def build_option():
    parser = OptionParser()
    parser.add_option("--skip-metrics", action="store_true", default=False,
                      help="Skip computing metrics on input data")
    parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")
    return parser


def main():
    parser = build_option()
    (options, args) = parser.parse_args()



if __name__ == '__main__':
    main()