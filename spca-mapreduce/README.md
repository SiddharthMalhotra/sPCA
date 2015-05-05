sPCA-mapreduce
===========

sPCA-mapreduce is a scalable implementation of Principal component analysis (PCA) algorithm on top of MapReduce. sPCA-mapreduce has been tested on Hadoop 0.20.204 & Hadoop 2.4.0 on the Linux OS. In the following, we will take you through running PCA on a toy matrix. We will run the example on the local machine without the need for setting up a cluster.

Download and Install Hadoop
==========================

Download and setup a single node hadoop cluster by following the instructions in [this link](http://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-common/SingleCluster.html). 

Clone the sPCA repo
==========================
Open the shell and clone the sPCA github repo:
```
git clone https://github.com/Qatar-Computing-Research-Institute/sPCA
```
In order to build sPCA-mapreduce source code, you need to install maven. You can download and install maven by folliwng this [quick tutorial] (http://maven.apache.org/guides/getting-started/maven-in-five-minutes.html). To verify that maven is installed, run the following 
command:
```
mvn --version
```
It should print out your installed version of Maven. After that, you can build sPCA by typing:

```
cd sPCA/spca-mapreduce
mvn package
```
To make sure that the code is build successfully, You should see something like the following:
```
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
```
Moreover, you will find a .jar file generated under `sPCA/spca-mapreduce/target/mapreducePCA-1.0.jar`. This jar file will be used to run the example in the following step.

Input/Output Format
=====================================
sPCA accepts the input matrix and writes the output in the Hadoop [SequenceFile](http://hadoop.apache.org/docs/r2.6.0/api/org/apache/hadoop/io/SequenceFile.html) format. 

Running sPCA in the local mode
=====================================
The next step is to run sPCA-mapreduce on a small toy matrix. There is an example script located in `sPCA/spca-mapreduce/spca-mapreduce_example.sh`. First, you need to set the environment variable `HADOOP_HOME` to the directory where Hadoop is downloaded and installed:
```
export HADOOP_HOME=<path/to/hadooop/directory> (e.g., /usr/lib/hadoop-2.4.0)
```
Then you need to copy the input matrix to the Hadoop distributed filesystem (HDFS), using the following command
```
$HADOOP_HOME/bin/hadoop fs -copyFromLocal sPCA/spca-mapreduce/input/seqfiles/ hdfs:///user/<username>
```
You can then run the example through the following command:
```
sPCA/spca-mapreduce/spca-mapreduce_example.sh
```
The output will be written in the hdfs directory `hdfs:///user/<username>/output`. The example involves a command similar to the following:
```
hadoop jar target/mapreducePCA-1.0-job.jar org.qcri.pca.SPCADriver \
-i <path/to/input/matrix/on/hdfs> -o <path/to/outputfolder/on/hdfs> -rows <number of rows> -cols <number of columns> -pcs <number of principal components> [-errSampleRate=<Error sampling rate>] [-maxIter=<max iterations>] [-normalize<0/1 (normalize input matrix or not)>]
```
This command runs sPCA-mapreduce on top of MapReduce in the local machine. The following is a description of the command-line arguments of sPCA:
- ```<path/to/input/matrix/on/hdfs>: ```Hdfs directory that contains an example input matrix in the sequenceFileFormat <IntWritable key, VectorWritable value>.
- ```<path/to/outputfolder/on/hdfs>:``` Hdfs directory where the resulting principal components is written
- ```<number of rows>:``` Number of rows for the input matrix
- ```<number of columns>:``` Number of columns for the input matrix
- ```<number of principal components>:``` Number of desired principal components
- ```[<Error sampling rate>](optional):``` The error sampling rate [0-1] that is used for computing the error, It can be set to 0.01 to compute the error for only a small sample of the matrix, this speeds up the computations significantly)
- ```[<max iterations>] (optional):``` Maximum number of iterations before terminating, the default is 3
- ```<0/1 (normalize input matrix or not)>](optional)``` : 0 or 1 values that specifies whether the input matrix needs to be normalized or not. 1 means that the matrix should be normalized, 0 means that matrix should not be normalized. Normalization is done by dividing each column by (column_max-column_min).
