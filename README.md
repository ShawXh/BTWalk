# BTWalk
This is the implementation of TKDE20 paper ''[BTWalk: Branching Tree Random Walk for Multi-order Structured Network Embedding](https://doi.org/10.1109/TKDE.2020.3029061)'' (a copy is in this repo.). All of the parameters and the corresponding descriptions in `btvec.cpp`, `btvec.py`, `vec4cross.py` can be found in Table 2 in the paper.

## Requirements
- gsl c++ library
To install gsl lib on your own machine, you can refer to this [link](https://blog.csdn.net/u012248802/article/details/80655902) (in Chinese).

## Single-network embedding (BTVec)
To run network embedding for a single network (weighted and directed), you need a network file named `$name$-net.txt`, whose format is:
```
nid1 nid2 weight1
nid2 nid1 weight2
...
```
where `nid` should be integer and start from `0`.

To get embedding, you need to run:
```
python2 btvec.py --input $name$-net.txt --dim 128
```
and the embedding will be output at `./emb/$name$-128.txt`. 

For more hyper-parameters to control the training procedure, please refer to our paper and `btvec.py` and `btvec.cpp`.

## Cross-network embedding (Vec4Cross)
To run network embedding for crossing networks, you need two networks `${net1}$-net.txt` and `${net2}$-net.txt`, a partial alignment file `${net1}$2${net2}$_${portion}$`, and a ground-truth alignment file `${net1}$2${net2}$_gt`.. The format of network files is the same as the above. The format of alignment files `${net1}$2${net2}$_${portion}$` and `${net1}$2${net2}$_gt` is:
```
nid_in_net1 nid_in_net2
...
```

For example, to get the embedding, you may run:
```
python2 vec4cross.py --net1 ./networks/f2t/fb-net.txt --net2 ./networks/f2t/tt-net.txt --seed ./networks/f2t/f2t_30 --gt ./networks/f2t/f2t_gt --name-dis ./networks/f2t/edit_dist_matrix.pickle
```

The embedding will be output at `./emb/${net1}$-${net2}$-emb-128`.

Notes: the gamma used in the code is actually 1-gamma in the paper.

## Reference
If you use the code, please cite:
```
@ARTICLE{btwalk,
  author={Xiong, Hao and Yan, Junchi},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={BTWalk: Branching Tree Random Walk for Multi-Order Structured Network Embedding}, 
  year={2022},
  volume={34},
  number={8},
  pages={3611-3628},
  doi={10.1109/TKDE.2020.3029061}}
```

If you use the data of crossing networks, facebook-twitter and douban-weibo, please cite:
```
@inproceedings{Cao2016ASNets,
author = {Cao, Xuezhi and Yu, Yong},
title = {ASNets: A Benchmark Dataset of Aligned Social Networks for Cross-Platform User Modeling},
year = {2016},
isbn = {9781450340731},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/2983323.2983864},
doi = {10.1145/2983323.2983864},
booktitle = {Proceedings of the 25th ACM International on Conference on Information and Knowledge Management},
pages = {1881–1884},
numpages = {4},
keywords = {benchmark dataset, network alignment, user modeling},
location = {Indianapolis, Indiana, USA},
series = {CIKM ’16}
}
```
