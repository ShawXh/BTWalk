import sys
import argparse
import os
from gensim.models import Word2Vec
from gensim.models.word2vec import Word2VecVocab
from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='')
    return parser.parse_args()

args = parse_args()

print "======================================================="
print "preprocessing %s..." % args.input

name = args.input.strip().split("/")[-1]

net = {}
degree = {}

def build_net():
    print "building net..."
    if os.path.exists("tmpfile/%s-net.txt" % name):
        print "net file exists"
        return
    with open(args.input, "r") as fi:
        line = fi.readline()
        while line != "":
            ls = line.strip().split(" ")
            v1, v2 = map(int, ls[:2])
            net[v1] = {}
            net[v2] = {}
            degree[v1] = 0.0
            degree[v2] = 0.0
            line = fi.readline()
    with open(args.input, "r") as fi:
        line = fi.readline()
        while line != "":
            ls = line.strip().split(" ")
            v1, v2 = map(int, ls[:2])
            w = float(ls[2])
            net[v1][v2] = w
            degree[v1] += w
            line = fi.readline()
#    assert(max(degree.keys())==len(degree), "node ids should start from 0")

def write_net():
    if os.path.exists("tmpfile/%s-net.txt" % name):
        print "net file exists"
        return
    print "writing net..."
    max_id = 0
    for i in net:
        max_id = max(max_id, i)
    with open("tmpfile/%s-net.txt" % name, "w") as fo:
        fo.write("%d\n" % (max_id + 1))
        for v1 in net:
            for v2 in net[v1]:
                fo.write("%d %d %f\n" % (v1, v2, net[v1][v2]))

def write_degree():
    if os.path.exists("tmpfile/%s-degree.txt" % name):
        print "degree file exists"
        return
    print "writing degree..."
    with open("tmpfile/%s-degree.txt" % name, "w") as f:
        for v in net:
            f.write("%d %f\n" % (v, degree[v]))

build_net()
write_net()
write_degree()
print "======================================================="
