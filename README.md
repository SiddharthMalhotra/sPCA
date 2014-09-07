ScalablePCA
===========

Scalable PCA (sPCA) is a scalable implementation of Principal component analysis (PCA) algorithm on top of Spark. sPCA has been tested on Apache Spark 1.0.0 and Linux OS. It should work with more recent Spark versions with only minor modifications; however, switching to another platform (e.g., Mac) will require recompiling the jars. In the following, we will take you through running PCA on a toy matrix. We will use Spark local mode which does not require setting up a cluster.

Download and Install Spark
==========================

Download  Spark 1.0.0+ [here](https://spark.apache.org/downloads.html). We refer to the directory where spark is downloaded by `SPARK_HOME`. After Spark is downloaded, build it using the following command:
```
SPARK_HOME/sbt/sbt assembly
```

You can also build Spark using Maven by following [this] (http://spark.apache.org/docs/1.0.0/building-with-maven.html) tutorial.

Verify that Spark is running by executing the SparkPi example. In the shell, run the following command:
```
SPARK_HOME/bin/run-example SparkPi 10
```
After the above Spark local program finishes, you should see the computed value of pi (something that's reasonably closer to 3.14).

Clone the ScalablePCA repo
==========================
Open the shell and clone the ScalablePCA github repo:
```
git clone https://github.com/Qatar-Computing-Research-Institute/ScalablePCA
```
In order to build sPCA source code, you need to install maven. You can download and install maven by folliwng this [quick tutorial] (http://maven.apache.org/guides/getting-started/maven-in-five-minutes.html). To verify that maven is installed, run the following 
command:
```
mvn --version
```
It should print out your installed version of Maven. After that, you can build sPCA by typing:

```
cd ScalablePCA/
mvn package
```
To make sure that the code is build successfully, You should see something like the following:
```
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
```
Moreover, you will find a .jar file generated under `ScalablePCA/target/SparkPCA.jar`. This jar file will be used to run the example in the following step.

Running ScalablePCA in the local mode
=====================================
The next step is to run sPCA on a small toy matrix. There is an example script located in `ScalablePCA/spca-example.sh`. you can run it through the following command:
```
./ScalablePCA/spca-example.sh
```
The example involves a command similar to the following:
```
SPARK_HOME/bin/spark-submit --class org.qcri.sparkpca.SparkPCA --master <master-url> target/sparkPCA-1.0.jar  <path/to/input/matrix> <path/to/outputfile> <number of rows> <number of columns> <number of principal components> [<Error sampling rate>] [<max iterations>]
```
This command runs the main class in the local mode using the following parameters in the following order:
- `<master-url>: `The master URL for the cluster (e.g. spark://23.195.26.187:7077), it is set to local for running locally local mode 
-	`<path/to/input/matrix>:` directory that contains an example input matrix in the sequenceFileFormat `<IntWritable key, VectorWritable value>`.
-	`<path/to/outputfile>:` The file where the resulting principal components is written
-	`<number of rows>:` Number of rows for the input matrix 
-	`<number of columns>:` Number of columns for the input matrix : 5 
-	`<number of principal components>:` Number of desired principal components 
-	`[<Error sampling rate>](optional):` The error sampling rate [0-1] that is used for computing the error, It can be set to 0.01 to compute the error for only a small sample of the matrix, this speeds up the computations significantly 
- `[<max iterations>] (optional):` Maximum number of iterations before terminating, the default is 3 

Running ScalablePCA on amazon ec2 cluster
=========================================


