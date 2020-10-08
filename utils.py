import os
import numpy as np
from copy import deepcopy
import random

def _pre(raw_net):
    if not os.path.exists("./tmpfile"):
        os.system("mkdir tmpfile")
    os.system("python2 build-my-net.py --input %s" % raw_net)
    if not os.path.exists("./btvec"):
        os.system("sh compile.sh btvec.cpp btvec")
    if not os.path.exists("./plus"):
        os.system("sh compile.sh plus.cpp plus")
    if not os.path.exists("./norm"):
        os.system("sh compile.sh norm.cpp norm")
    return
    
def _train(net_path, emb_prefix, degree_path, conf_path,\
           sample=100, dim=64, alpha=0.1, beta=0.5, ns=5, K=1, Nm=1, load=0, tp=0, link=0, binary=0, norm=1, thread=16, w=0.2, decay=0.5):
    print "=============================================="
    os.system("./btvec -input %s -type %d -link %d -binary %d -samples %d -emb-u %s -emb-v %s -dim %d -alpha %f -beta %f -degree %s -conf %s -negative %d -H %d -topk %d -load %d -thread %d -w %f -decay %f"\
              % (net_path, 
                 tp, link, binary, sample, 
                  emb_prefix + "-%d-u-wo-norm-%d" % (tp+1, dim), 
                  emb_prefix + "-%d-v-wo-norm-%d" % (tp+1, dim), 
                  dim, alpha, beta,
                  degree_path, conf_path, 
                  ns, K, Nm, load, thread,
                  w, decay))
    if (norm == 1):
        os.system("./norm -input %s -output %s -binary 0"\
                  % (emb_prefix + "-%d-u-wo-norm-%d" % (tp+1, dim), emb_prefix + "-%d-u-%d" % (tp+1, dim)))
        
    return

def get_cand_from_align(net1, net2, align, new_align=None, new_conf=None):
    align_ = dict(zip(align.values(), align.keys()))

    t_align = deepcopy(align)
    t_align_ = deepcopy(align_)

    if new_align:
        new_align = dict(zip(new_align.values(), new_align.keys()))
        t_align.update(new_align)
        t_align_.update(new_align_)

    cand = []
    for i in t_align:
        j = t_align[i]
        for c1 in net1[i]:
            if c1 in t_align: continue
            for c2 in net2[j]:
                if c2 in t_align_: continue
                cand.append((c1, c2))

    if new_conf:
        for i in new_conf:
            for j in new_conf[i]:
                for c1 in net1[i]:
                    if c1 in align: continue
                    for c2 in net2[j]:
                        if c2 in align_: continue
                        cand.append((c1, c2))
   
    cand = list(set(cand))
    return cand

def cos_simi(arr1, arr2):
    return arr1.dot(arr2)/(1e-6 + np.sqrt(arr1.dot(arr1)*arr2.dot(arr2)))

def exp_simi(arr1, arr2):
    delt = arr1 - arr2
    return np.exp(-delt.dot(delt))

def jcd_simi(net1, net2, v1, v2, sconf):
    n1 = net1[v1]
    n2 = net2[v2]

    na = 0
    for c1 in n1:
        if c1 in sconf:
            for c2 in sconf[c1]:
                if c2 in n2:
                    na += 1
                    break

    lc = len(n1) + len(n2) - na

    return 1.0 * na / lc 

def turn_2_sum_net(net_path, align):
    net = {}
    with open(net_path, "r") as f:
        line_1st = f.readline()
        line = f.readline()
        while line != "":
            ls = line.strip().split(" ")
            v1, v2 = map(int, ls[:2])
            try:
                net[v1][v2] = None
            except:
                net[v1] = {v2: None}
            try:
                net[v2][v1] = None
            except:
                net[v2] = {v1: None}
            line = f.readline()
            
    offset = len(net) / 2
    
    def map_f(vid):
        if vid >= offset and vid in align:
            return align[vid]
        else:
            return vid
    
    with open(net_path, "w") as f:
        f.write(line_1st)
        for i in net:
            for j in net[i]:
                f.write("%d %d 1\n" % (map_f(i), map_f(j)))
    
    return

def test_pr(res, pa, gt):
    '''pa: partial alignment, a dict
    gt: ground truth, a dict'''
    right = 0
    
    pa_conv = dict(zip(pa.values(), pa.keys()))
    new_res = {}
    
    for i in pa:
        new_res[i] = pa[i]
    for i in res:
        j = res[i]
        if i not in pa and j not in pa_conv:
            new_res[i] = j

    for i in new_res:
        if i not in pa:
            try:
                if gt[i] == res[i]:
                    right += 1
            except:
                pass
            
    a1 = len(new_res)-len(pa)
    a2 = len(gt)-len(pa)
    prec = float(right)*100.0/a1
    recall = float(right)*100.0/a2
    f1 = 2 * prec * recall / (prec + recall + 1e-5)
    print "Precision: [%d/%d] = %.3f" % (right, a1, prec)
    print "Recall: [%d/%d] = %.3f" % (right, a2, recall)
    print "F1: = %.3f" % f1
    
    return prec, recall, f1

def count_prec_N(s, gt, pa):
    '''latest version'''
    N=[1,5,10,15,20,25,30]
    right = [0] * 7
    right_ = [0] * 7
    
    pa_ = dict(zip(pa.values(), pa.keys()))
    gt_ = dict(zip(gt.values(), gt.keys()))
    
    n1, n2 = s.shape
    for i in gt:
        if i in pa:
            continue
        s_gt = s[i, gt[i]]
        if s_gt == 0:
            continue
        count = 0
        for j in xrange(n2):
            if j in pa_:
                continue
            if s[i, j] > s_gt:
                count += 1
        for n in xrange(7):
            if count < N[n]:
                right[n] += 1
    
    for i in gt_:
        if i in pa_:
            continue
        s_gt = s[gt_[i], i]
        if s_gt == 0:
            continue
        count = 0
        for j in xrange(n1):
            if j in pa:
                continue
            if s[j, i] > s_gt:
                count += 1
        for n in xrange(7):
            if count < N[n]:
                right_[n] += 1
                
    res = []
    for n in xrange(7):
        N_ = N[n]
        p = 50.0 * (float(right[n]) + float(right_[n])) / (len(gt)-len(pa))
        res.append(p)
        print "Prec@%d: [%d+%d/%d] = %.3f" % (N_, right[n], right_[n], len(gt)-len(pa), p)
        
    return res


def count_PR_N(s, gt, pa, conf, conf_, N):
    '''latest version'''
    right = 0 
    right_ = 0
    
    pa_ = dict(zip(pa.values(), pa.keys()))
    gt_ = dict(zip(gt.values(), gt.keys()))
    
    nodes = {}
    nodes_ = {}
    for i in conf: 
        nodes[i] = None
        for j in conf[i]:
            nodes_[j] = None
    for j in conf_:
        nodes_[j] = None
        for i in conf_[j]:
            nodes[i] = None
    
    n1, n2 = s.shape
    for i in nodes:
        if i in pa:
            continue
        s_gt = s[i, gt[i]]
        if s_gt <= 0:
            continue
        count = 0
        for j in xrange(n2):
            if j in pa_:
                continue
            if s[i, j] > s_gt:
                count += 1
        if count < N:
            right += 1
    
    for i in nodes_:
        if i in pa_:
            continue
        s_gt = s[gt_[i], i]
        if s_gt == 0:
            continue
        count = 0
        for j in xrange(n1):
            if j in pa:
                continue
            if s[j, i] > s_gt:
                count += 1
        if count < N:
            right_ += 1

    m = len(nodes) + len(nodes_)
    n = len(gt) - len(pa)
    p = 100.0 * (right + right_) / m
    r = 50.0 * (right + right_) / n
    print "Prec@%d: [%d+%d/%d] = %.3f" % (N, right, right_, m, p)
    print "Recall@%d: [%d+%d/%d] = %.3f" % (N, right, right_, n * 2, r)
        
    return p, r

