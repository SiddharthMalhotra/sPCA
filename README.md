sPCA
===========
Scalable PCA (sPCA) is a scalable implementation of Principal component analysis (PCA) on of Spark and MapReduce. sPCA achieves scalability via employing efficient large matrix operations, effectively leveraging matrix sparsity, and minimizing intermediate data. The repository contains two README files that will take you through running sPCA on Spark and MapReduce, respectively: ([sPCA-Spark README](spca-spark/README.md), [sPCA-mapreduce README](spca-mapreduce/README.md)).

Resources
==========================
- [arXiv technical report](http://arxiv.org/abs/1503.05214): Analysis of different methods for performing PCA and their limitations
                                                             in handling large-scale datasets on distributed clusters.
- [SIGMOD paper](http://ds.qcri.org/images/profile/tarek_elgamal/sigmod2015.pdf): Comparison with the closest PCA implementations, Mahout/MapReduce and MLlib/Spark.

License
==========================
sPCA is released under the terms of the [MIT License](http://opensource.org/licenses/MIT).

Contact
==========================
For any issues or enhancement please use the [issue pages](https://github.com/Qatar-Computing-Research-Institute/sPCA/issues) in Github, or [contact us](mailto:tarek.elgamal@gmail.com). We will try our best to help you sort it out.
