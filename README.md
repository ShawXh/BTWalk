# BTWalk
Notes: All of the parameters and the corresponding descriptions in btvec.cpp, btvec.py, vec4cross.py can be found in Table 2 in the paper.

## Single-network embedding (BTVec)
To run network embedding for a single network, you need a network file named `$name$-net.txt`, whose format is:
```
nid1 nid2
nid2 nid1
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
To run network embedding for crossing networks, you need two network
