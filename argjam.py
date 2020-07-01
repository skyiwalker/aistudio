
import argparse
if __name__ == '__main__':
    # Arguments Parsing
    parser = argparse.ArgumentParser(description='Test ArgParser')
    parser.add_argument('--test-arg1', type=int, default=1, metavar='N',
                        help='input integer(default: 1)')
    parser.add_argument('--test-arg2', type=int, default=2, metavar='N',
                        help='input integer(default: 2)')
    parser.add_argument('--test-arg3', type=int, metavar='N',
                        help='input integer')
    args = parser.parse_args()
    
    print(args.test_arg1)
    print(args.test_arg2)
    print(args.test_arg3)
