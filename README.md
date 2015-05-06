sPCA
===========

Scalable PCA (sPCA) is a scalable implementation of Principal component analysis (PCA) algorithm. sPCA achieves scalability via employing efficient large matrix operations, effectively leveraging matrix sparsity, and minimizing intermediate data. sPCA is implemented on top of Spark and MapReduce frameworks. The repository contains two README files that will take you through running sPCA on Spark and MapReduce: ([sPCA-Spark readme](spca-spark/README.md), [sPCA-mapreduce readme](spca-mapreduce/README.md))

Resources
==========================

- [arXiv technical report](http://arxiv.org/abs/1503.05214): Analysis of different methods for performing PCA and their limitations in handling large-scale datasets on distributed clusters
- [SIGMOD paper](http://ds.qcri.org/images/profile/tarek_elgamal/sigmod2015.pdf): Comparison with the closest PCA implementations, Mahout/MapReduce and MLlib/Spark
