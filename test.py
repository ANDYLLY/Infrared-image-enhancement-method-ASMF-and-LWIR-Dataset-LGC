import argparse
import ASMF

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()

    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--Scale', default=6, type=int)
    parser.add_argument('--Mdf_Epochs', default=50, type=int)
    parser.add_argument('--Dataset', required=True, type=str)

    args = parser.parse_args()

    model = ASMF.ASMF(args.name, args.Scale, args.Mdf_Epochs, args.Dataset)
    model.test()