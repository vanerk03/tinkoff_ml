import argparse
import dill
from train import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the model")
    parser.add_argument("--prefix", type=str,
                        help="Start of the sentence", default=None)
    parser.add_argument("--length", type=int,
                        help="Length of the sequence to generate")
    args = parser.parse_args()

    with open(args.model, "rb") as fl:
        mdl: Model = dill.load(fl)

    print(mdl.generate(args.prefix, args.length))


if __name__ == "__main__":
    main()
