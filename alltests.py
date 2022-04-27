from importlib import import_module
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", action="store_true", dest="pickle", help="Generate pickle test files")
parser.add_argument("-c", action="store_true", dest="cupy", help="Run cupy test files")
args = parser.parse_args()

if args.pickle:
    print("Generate pickle test files ...")
    import_module("tests.pickle-test")
    print("Finished.")
else:
    if args.cupy:
        print("Running cupy tests ...")
        import_module("tests.cupy-test")
        print("Finished.")
    else:
        print("Running numba tests ...")
        import_module("tests.numba-test")
        print("Finished.")
