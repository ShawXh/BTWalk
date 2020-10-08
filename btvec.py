import os
import argparse
import time
from utils import _pre
start_time = time.time()

def parse_args():
    '''
    Running for Single-Net embedding.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', help="/data/blog-net.txt")
    parser.add_argument('--alpha', type=float, default=0.1, help="weight of the Laplacian Regularizer term")
    parser.add_argument('--beta', type=float, default=0.5, help="weight of alignment loss, for cross-net")
    parser.add_argument('--K', type=int, default=2, help="maximum hops of BTWalk")
    parser.add_argument('--sample', type=int, default=300, help="how many millions of samples in total")
    parser.add_argument('--ns', type=int, default=5, help="negative samples for each node")
    parser.add_argument('--dim', type=int, default=128, help="embedding dimension")
    parser.add_argument('--bin', type=int, default=1, help="1 for outputing embedding in the binary format; 0 otherwise")
    parser.add_argument('--link', type=int, default=0, help="0 for single net, 2 for cross net")
    parser.add_argument('--Nm', type=int, default=1, help="for cross-network setting, maximum number of estimated matching nodes for each node; omitting this arguments for single-network")
    parser.add_argument('--thread', type=int, default=16, help="number of parallel threads")
    return parser.parse_args()

args = parse_args()
name = args.input.strip().split("/")[-1]
prefix = "./tmpfile/"
if not os.path.exists("./tmpfile"):
    os.system("mkdir tmpfile")
net_path = prefix + "%s-net.txt" % name
degree_path = prefix + "%s-degree.txt" % name
if not os.path.exists("./emb"):
    os.system("mkdir emb")
emb_path = "emb/%s-emb" % name
conf_path = "tmpfile/%s-conf.txt" % name
sample = args.sample
alpha = args.alpha
beta = args.beta
link = args.link

# preprocessing
_pre(args.input)

# ## undirected info
print "=============================================="
os.system("./btvec -input %s -type 0 -link %d -binary 0 -samples %d -emb-u %s -emb-v %s -dim %d -alpha %f -beta %f -degree %s -conf %s -negative %d -K %d -Nm %d -thread %d"\
          % (net_path, 
              args.link, 
              args.sample, 
              emb_path + "-1-u-wo-norm-%d" % (args.dim/2), 
              emb_path + "-1-v-wo-norm-%d" % (args.dim/2), 
              args.dim/2, 
              args.alpha, 
              args.beta,
              degree_path, 
              conf_path, 
              args.ns, 
              args.K, 
              args.Nm,
              args.thread))
os.system("./norm -input %s -output %s -binary 0 -save-form 0"\
          % (emb_path + "-1-u-wo-norm-%d" % (args.dim/2), emb_path + "-1-u-%d" % (args.dim/2)))
#os.system("rm -rf %s" % emb_path + "-1-u-wo-norm-%d" % args.dim)

# ## directed info
print "=============================================="
os.system("./btvec -input %s -type 1 -link %d -binary 0 -samples %d -emb-u %s -emb-v %s -dim %d -alpha %f -beta %f -degree %s -conf %s -negative %d -K %d -Nm %d -thread %d"\
          % (net_path, 
              args.link, 
              args.sample, 
              emb_path + "-2-u-wo-norm-%d" % (args.dim/2), 
              emb_path + "-2-v-wo-norm-%d" % (args.dim/2), 
              args.dim/2, 
              args.alpha, 
              args.beta,
              degree_path, 
              conf_path, 
              args.ns, 
              args.K, 
              args.Nm,
              args.thread))
os.system("./norm -input %s -output %s -binary 0"\
          % (emb_path + "-2-u-wo-norm-%d" % (args.dim/2), emb_path + "-2-u-%d" % (args.dim/2)))
#os.system("rm -rf %s" % emb_path + "-2-u-wo-norm-%d" % args.dim)
# ## concat
print "=============================================="
if args.bin == 1:
    os.system("./plus -emb1 %s -emb2 %s -output %s -binary %d"\
          % (emb_path + "-1-u-%d" % (args.dim/2), emb_path + "-2-u-%d" % (args.dim/2), emb_path + "-%d.bin" % args.dim, args.bin))
else:
    os.system("./plus -emb1 %s -emb2 %s -output %s -binary %d"\
          % (emb_path + "-1-u-%d" % (args.dim/2), emb_path + "-2-u-%d" % (args.dim/2), emb_path + "-%d.txt" % args.dim, args.bin))

# ## time counting
end_time = time.time()
print "Used time: %f" % (end_time - start_time)
