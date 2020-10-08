import argparse
import numpy as np
from sklearn import svm
import pickle
import os
import sys
import time
from copy import deepcopy
import random

from utils import _pre, _train, get_cand_from_align, cos_simi, turn_2_sum_net, exp_simi, jcd_simi
from utils import count_prec_N, test_pr, count_PR_N

class Model(object):
    def __init__(self, 
                 net1, 
                 net2, 
                 seed, 
                 gt, 
                 name_dis,
                 dim, 
                 ns,
                 alpha, 
                 beta, 
                 gamma,
                 Nm, 
                 K,
                 Np,
                 sample, 
                 thread,
                 test_emb,
                 only_svm,
                 w,
                 decay,
                 epochs,
                ):
        self.dim = dim
        self.ns = ns
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.Nm = Nm
        self.K = K  # num of hops
        self.sample = sample
        self.n = Np
        self.thread = thread
        self.test_emb = test_emb
        self.osvm = only_svm
        self.w = w
        self.decay = decay
        self.epochs = epochs
        
        self.epoch = 0
        
        self._file_init(net1, net2, seed)
        self._net_init(net1, net2)
        self._align_init(seed, gt)
        self._name_init(name_dis)

        _pre(net1)
        _pre(net2)

    def _file_init(self, net1, net2, seed):
        # init file settings
        
        name = net1.strip().split("/")[-1][:-7] + \
                net2.strip().split("/")[-1][:-7]
        self.name = name
        if not os.path.exists("./tmpfile"):
            os.system("mkdir tmpfile")
        self.net_path = "tmpfile/%snet.txt" % name
        self.degree_path = "tmpfile/%sdegree.txt" % name
        if not os.path.exists("./emb"):
            os.system("mkdir emb")
        self.emb_prefix = "emb/%semb" % name
        self.emb_path = self.emb_prefix + "-%d" % self.dim
        if self.test_emb != "":
            self.emb_path = self.test_emb
        
        self.conf_path = "tmpfile/%sconf.txt" % name
        if not os.path.exists("./result"):
            os.system("mkdir result")
        if not os.path.exists("./result/%s" % name[:-1]):
            os.system("mkdir result/%s" % name[:-1])
        if not os.path.exists("./result/%s/gamma_%s_beta_%s" % (name[:-1], str(self.gamma), str(self.beta))):
            os.system("mkdir ./result/%s/gamma_%s_beta_%s" % (name[:-1], str(self.gamma), str(self.beta)))
        self.res_path = "result/%s/gamma_%s_beta_%s/%s-result_epoch" % (name[:-1], str(self.gamma), str(self.beta), seed[-2:])
        
        print "= =P=y=t=h=o=n = =F=i=l=e= =P=a=t=h= = ="
        print "Net Path:", self.net_path
        print "Dgr Path:", self.degree_path
        print "Emb Path:", self.emb_path
        print "Cnf Path:", self.conf_path
        print "= = = = = = = = = = = = = = = = = = = = ="
        
        return
         
    def _net_init(self, net1, net2):
        # building a self-contained network type.
        def read_net(net_path):
            net = {}
            with open(net_path, "r") as f:
                line = f.readline()
                while line != "":
                    l = line.strip().split(" ")
                    v1, v2 = map(int, l[:2])
                    w = float(l[2])
                    try:
                        net[v1][v2] = w
                    except:
                        net[v1] = {v2: w}
                    line = f.readline()
            return net

        # read net
        self.net1 = read_net(net1)
        self.n1 = max(self.net1.keys()) + 1
        self.net2 = read_net(net2)
        self.n2 = max(self.net2.keys()) + 1

        # building the new net
        offset = len(self.net1)
        net = {}
        for i in self.net1:
            net[i] = self.net1[i]
        for i in self.net2:
            net[i + offset] = map(lambda x: x + offset, self.net2[i])
        self.net = net

        # write new net
        with open(self.net_path, "w") as f:
            f.write("%d\n" % (self.n1 + self.n2))
            for i in net:
                for j in net[i]:
                    f.write("%d %d 1\n" % (i, j))

        with open(self.degree_path, "w") as f:
            for i in self.net:
                f.write("%d %d\n" % (i, len(self.net[i])))

        return
        
    def _name_init(self, name_dis):
        # initialize name information model, e.g. edit-distance here.
        if name_dis == "":
            self.name_dis = False
            return
        with open(name_dis, "r") as f:
            self.name_dis = pickle.load(f)
        
        return
       
    def _align_init(self, seed, gt):
        def read_align(file_path):
            align = {}  # net1 -> net2
            with open(file_path, "r") as f:
                for line in f.readlines():
                    v1, v2 = map(int, line.strip().split(" ")[:2])
                    align[v1] = v2
            return align

        # initialize alignments information
        self.raw_align = read_align(seed)
        self.raw_align_ = dict(zip(self.raw_align.values(),\
                self.raw_align.keys()))
        self.new_align = {}
        self.new_align_ = {}
        self.conf = {}
        self.conf_ = {}
        self.gt = read_align(gt)
        self.gt_ = dict(zip(self.gt.values(),\
                self.gt.keys()))
        
        # writing initial confidence
        self._write_conf()
        return
        
    def _train_emb(self):
        load = 1
        if self.epoch == 1:
            load = 0

        _train(
                net_path=self.net_path, 
                sample=self.sample, 
                dim=self.dim/2,
                alpha=self.alpha, 
                beta=self.beta, 
                ns=self.ns, 
                K=self.K, 
                Nm=self.Nm, 
                link=2, 
                tp=0,
                emb_prefix=self.emb_prefix, 
                degree_path=self.degree_path, 
                conf_path=self.conf_path,
                load=0, 
                thread=self.thread,
                w=self.w,
                decay=self.decay,
            )

        _train(
                net_path=self.net_path, 
                sample=self.sample, 
                dim=self.dim/2,
                alpha=self.alpha, 
                beta=self.beta, 
                ns=self.ns, 
                K=self.K, 
                Nm=self.Nm, 
                link=2, 
                tp=1,
                emb_prefix=self.emb_prefix, 
                degree_path=self.degree_path, 
                conf_path=self.conf_path,
                load=0, 
                thread=self.thread,
                w=self.w,
                decay=self.decay,
            )
        print "=============================================="
        os.system("./plus -emb1 %s -emb2 %s -output %s -binary 0"\
                      % (
                          self.emb_prefix + "-1-u-%d" % (self.dim/2), 
                          self.emb_prefix + "-2-u-%d" % (self.dim/2), 
                          self.emb_prefix + "-%d" % self.dim))

        return
            
    def _read_emb(self):
        with open(self.emb_path, "r") as f:
            N, d = map(int, f.readline().strip().split(" "))
            
            emb1 = np.zeros((self.n1, d))
            emb2 = np.zeros((self.n2, d))

            line = f.readline()
            while line!="":
                ls = line.strip().split(" ")
                vid = int(ls[0])
                if vid < self.n1:
                    emb1[vid] = np.array(map(float, ls[1:]))
                else:
                    emb2[vid - self.n1] = np.array(map(float, ls[1:]))

                line = f.readline()
                
        self.emb1 = emb1
        self.emb2 = emb2

        return
    
    def _write_conf(self):
        with open(self.conf_path, "w") as f:
            f.write("%d %d\n" % (self.n1, self.n2))
            for i in self.raw_align:
                j = self.raw_align[i] + self.n1
                f.write("%d %d 1\n" % (i, j))
                f.write("%d %d 1\n" % (j, i))

            if self.epoch == 0:
                pass
            else:
                try:
                    for i in self.conf:
                        for j in self.conf[i]:
                            j_ = j + self.n1
                            c = self.conf[i][j]
                            if c > 0:
                                f.write("%d %d %f\n" % (i, j_, c))
                    for j in self.conf_:
                        j_ = j + self.n1
                        for i in self.conf_[j]:
                            c = self.conf_[j][i]
                            if c > 0:
                                f.write("%d %d %f\n" % (j_, i, c))
                except:
                    print "Warning: pass writing conf."
        return

    def _cal_jcd(self):
        def _build_super_conf():
            sconf = {}
            for i in self.raw_align:
                sconf[i] = {self.raw_align[i]: 1}
            if self.conf:
                for i in self.conf:
                    for j in self.conf[i]: 
                        try:
                            sconf[i][j] = self.conf[i][j]
                        except:
                            sconf[i] = {j: self.conf[i][j]}
                for j in self.conf_:
                    for i in self.conf_[j]:
                        try:
                            sconf[i][j] = self.conf_[j][i]
                        except:
                            sconf[i] = {j: self.conf_[j][i]}
            return sconf
        sconf = _build_super_conf()


        jcds = np.zeros((self.n1, self.n2))
        for v1 in self.net1:
            for v2 in self.net2:
                s = jcd_simi(net1 = self.net1, 
                        net2 = self.net2, 
                        v1 = v1, v2 = v2,
                        sconf = sconf)
                jcds[v1, v2] = s
        self.jcds = jcds
        return

    def _cal_cn(self, align, gamma):
        align_ = dict(zip(align.values(), align.keys()))

        cn = (np.ones((self.n1, self.n2)) + 0.00001 * np.random.random((self.n1, self.n2))) * gamma
        for i in align:
            j = align[i]
            if i not in self.net1 or j not in self.net2:
                continue
            for c1 in self.net1[i]:
                if c1 in align: continue
                for c2 in self.net2[j]:
                    if c2 in align_: continue
                    cn[c1, c2] = 1

        return cn

    def _train_svm(self):
        self._cal_jcd()

        self.classifier = svm.SVC(C=1.0, kernel="linear", probability=True)

        def _generate_train_data():
            self.raw_pos_pair = zip(self.raw_align.keys(), self.raw_align.values())
            self.pos_pair = self.raw_pos_pair
            
            if self.epoch > 1:
                self.new_pos_pair = zip(self.new_align.keys(), self.new_align.values())
                self.pos_pair += self.new_pos_pair

            def _gen_neg_pair(num_neg_pairs=len(self.net1)):
                # random match
                v1 = random.sample(self.net1.keys(), num_neg_pairs)
                v2 = random.sample(self.net2.keys(), num_neg_pairs)
                return zip(v1, v2)

            def f(pair):
                return [self.jcds[pair[0], pair[1]], self.name_dis[pair[0], pair[1]]]
            
            X_pos = map(f, self.pos_pair)
            X_neg = map(f, _gen_neg_pair())

            X = X_pos + X_neg
            y = [1] * len(X_pos) + [0] * len(X_neg)

            return X, y

        X, y = _generate_train_data()
        self.classifier.fit(X, y)
        return

    def _cal_svm_score(self):
        pairs = [(v1, v2) for v1 in self.net1 for v2 in self.net2]
        def f(pair):
            return [self.jcds[pair[0], pair[1]], self.name_dis[pair[0], pair[1]]]
        X = map(f, pairs)

        y = self.classifier.predict_proba(X)[:, 1]

        svms = 0.00001 * np.random.random((self.n1, self.n2))
        for i, pair in enumerate(pairs):
            v1, v2 = pair
            svms[v1, v2] = y[i]

        return svms

    def _cal_embscore(self, align):
        # cosine similarity
        align_ = dict(zip(align.values(), align.keys()))
        scores = -1 * np.ones((self.n1, self.n2)) + 0.00001 * np.random.random((self.n1, self.n2))
        for c1 in self.net1:
            for c2 in self.net2:
                s = cos_simi(self.emb1[c1], self.emb2[c2])
                scores[c1, c2] = s
        
        return scores
        
    def _do_align(self):
        new_conf_num = self.n * self.epoch
        print "max new conf num:", new_conf_num
       
        self.cn1 = self._cal_cn(self.raw_align, self.gamma)  # gamma = 0.95

        if not self.osvm:
            print "Scoring embedding..."
            self.embs = self._cal_embscore(self.raw_align)
            self.S = (self.embs + 1) * 0.5 

        if (self.name_dis is not False) or (self.osvm):
            print "SVM training..."
            self._train_svm()
            print "SVM scoring..."
            self.svms = self._cal_svm_score()

            if self.osvm:
                self.S = self.svms
            else:
                self.S = np.max(np.stack((self.embs, self.svms * 0.8 + 0.2)), axis=0)
        
        self.S *= self.cn1  # for score

        ## Add confidence
        scores = deepcopy(self.S)
        s = scores.reshape(1, -1)[0]
        res = s.argsort()
        l = res.shape[0]

        conf = {}
        conf_ = {}

        new_align = {}

        i = l - 1
        while len(conf) < new_conf_num and len(conf_) < new_conf_num:
            m = res[i] / self.n2
            n = res[i] % self.n2

            if m in self.raw_align or n in self.raw_align_:
                i -= 1
                if i == -l: break
                continue

            if m not in new_align:
                new_align[m] = n

            if m not in conf:
                conf[m] = {}
            if len(conf[m]) < self.Nm:
                conf[m][n] = s[i]

            if n not in conf_:
                conf_[n] = {}
            if len(conf_[n]) < self.Nm:
                conf_[n][m] = s[i]

            i -= 1

            if i == -l:
                break

        self.new_align = new_align  # for svm training. no other use.
        print "new align num:", len(self.new_align)
        self.conf = conf
        self.conf_ = conf_

        return

    def _assess(self):
        pr = test_pr(res = self.new_align,
                pa = self.raw_align,
                gt = self.gt)

        print "=====  total result:"
        precN1= count_prec_N(
                s = self.S,
                gt = self.gt,
                pa = self.raw_align)
        
        print "===== partial result:"
        PR = count_PR_N(
                s = self.S,
                gt = self.gt,
                pa = self.raw_align,
                conf = self.conf,
                conf_ = self.conf_,
                N = 5)

        result = {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "Nm": self.Nm,
                "pr": PR,
                "precN": precN1}
        
        with open(self.res_path + "_%d.pkl" % self.epoch, "w") as f:
            pickle.dump(result, f)
        with open(self.res_path + "_%d_S.pkl" % self.epoch, "w") as f:
            pickle.dump(self.S, f)
                
        return
    
    def train_epoch(self):
        self.epoch += 1
        print "################"
        print "##   epoch %d  ##" % self.epoch
        print "################"
        if self.test_emb == "" and (not self.osvm):
            print "Training..."
            self._train_emb()
        if not self.osvm:
            print "Reading Embedding..."
            self._read_emb()
        print "Begin aligning..."
        self._do_align()
        print "Assessing..."
        self._assess()
        if self.test_emb:
            sys.exit(0)
        print "Writing confidence..."
        self._write_conf()

        return
    
    def train_all(self):
        while len(self.new_align) < (len(self.gt) - len(self.raw_align)):
            self.train_epoch()

            if self.epoch == self.epochs:
                sys.exit()
        return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net1', type=str, default='')
    parser.add_argument('--net2', type=str, default='')
    parser.add_argument('--seed', type=str, default='')
    parser.add_argument('--gt', type=str, default='')
    parser.add_argument('--name-dis', type=str, default='', help="path of the files recoding edit distances between user names")
    
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--w', type=float, default=0.3)
    parser.add_argument('--decay', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--sample', type=int, default=500)
    parser.add_argument('--thread', type=int, default=16)
    parser.add_argument('--ns', type=int, default=5)
    parser.add_argument('--Np', type=int, default=9999, help="maximum number of newly estimated matchings in an EM iteration")
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--Nm', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=1, help="number of EM iterations, 1 means actually only adopts the estimation step.")
    parser.add_argument('--test-emb', type=str, default="", help="embedding path (from other embedding methods, e.g. deepwalk) to be evaluated on the network alignment task")
    parser.add_argument('--only-svm', action="store_true", default=False, help="only test the embedding by an svm classifier on the handcrafted features (username edit distances & jaccard similarity)")

    return parser.parse_args()
        
if __name__ == "__main__":
    args = parse_args()
    
    model = Model(
        net1=args.net1, 
        net2=args.net2,
        seed=args.seed, 
        gt=args.gt, 
        name_dis=args.name_dis,
        dim=args.dim, 
        ns=args.ns, 
        alpha=args.alpha, 
        beta=args.beta, 
        gamma=args.gamma, 
        Nm=args.Nm, 
        K=args.K, 
        Np=args.Np,
        sample=args.sample, 
        thread=args.thread, 
        test_emb=args.test_emb,
        only_svm=args.only_svm,
        w=args.w,
        decay=args.decay,
        epochs=args.epochs,
    )

    model.train_all()
#    model.train_epoch()
