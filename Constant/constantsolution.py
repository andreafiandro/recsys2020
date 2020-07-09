from constantutils import ConstantOptimization
import argparse

parser = argparse.ArgumentParser(description='POLINKS Optimized constant solution for Recsys Challenge 2020')
parser.add_argument('--trainingpath', type=str, help='Define the path for the file training.tsv', default='./training.tsv')
parser.add_argument('--validationpath', type=str, help='Define the path for the file val.tsv', default='./val.tsv')
parser.add_argument('--testpath', type=str, help='Define the path for the file test.tsv', default='./competition_test.tsv')
parser.add_argument('--computectr', dest='computectr', action='store_true')
parser.set_defaults(computectr=False)

args = parser.parse_args()

coAlgorithm = ConstantOptimization(args.trainingpath)
coAlgorithm.optimize_constant(args.validationpath, 'validation', compute_ctr=args.computectr)
coAlgorithm.optimize_constant(args.testpath, 'test', compute_ctr=args.computectr)
