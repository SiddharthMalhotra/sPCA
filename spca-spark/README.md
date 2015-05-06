sPCA-spark
===========

sPCA-spark is a scalable implementation of Principal component analysis (PCA) on top of Spark. sPCA has been tested on Apache Spark 1.0.0. In the following, we will take you through running sPCA-spark on a toy matrix. We will use Spark local mode which does not require setting up a cluster.

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
After the above Spark local program finishes, you should see the computed value of pi (something that's reasonably close to 3.14).

Clone the sPCA repo
==========================
Open the shell and clone the sPCA github repo:
```
git clone https://github.com/Qatar-Computing-Research-Institute/sPCA
```
In order to build sPCA source code, you need to install maven. You can download and install maven by folliwng this [quick tutorial] (http://maven.apache.org/guides/getting-started/maven-in-five-minutes.html). To verify that maven is installed, run the following 
command:
```
mvn --version
```
It should print out your installed version of Maven. After that, you can build sPCA-spark by typing:

```
cd sPCA/spca-spark
mvn package
```
To make sure that the code is build successfully, You should see something like the following:
```
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
```
Moreover, you will find a .jar file generated under `sPCA/spca-spark/target/SparkPCA.jar`. This jar file will be used to run the example in the following step.

Input Format
=====================================
sPCA-spark accepts the input matrix in the Hadoop [SequenceFile](http://hadoop.apache.org/docs/r2.6.0/api/org/apache/hadoop/io/SequenceFile.html) format. However, sPCA-spark provides an auxiliary class to convert two other formats in to the SequenceFile format. The two formats are:
- `Dense Matrix Format:` We refer to this format as `DENSE`. DENSE stores one line for each row and the values in each row are separated by white spaces. The input can be one file or a directory with multiple files.
- `Coordinate List Format:` We refer to this format as `COO`. COO stores a list of (row, column, value) tuples. The indices are sorted by row index then column index. The input can be one file or a directory with multiple files.

The repository contains some examples from both formats under the directory `sPCA/spca-spark/input`. The following shows an example of how to convert them to the SequenceFile format:
```
java -classpath target/sparkPCA-1.0.jar -DInput=input/mat.txt -DInputFmt=DENSE -DOutput=output/dense -DCardinality=5 org.qcri.sparkpca.FileFormat

java -classpath target/sparkPCA-1.0.jar -DInput=input/coo_0-based.txt -DInputFmt=COO -DOutput=output/coo_0-based -DBase=0 -DCardinality=5 org.qcri.sparkpca.FileFormat
```
The parameters are described as follows:
- `-DInput:` File or directory that contains the matrix that will be converted.
- `-DInputFmt:` Format of the input matrix. As described above, there are two supported formats (DENSE & COO).
- `-DOutput:` Output folder where the converted matrix will be written.
- `-DBase:` 0 or 1 values that specifies whether the row and column indices are 0-based or 1-based.
- `-DCardinality:` The number of columns of the input matrix.


Output Format
=====================================
sPCA supports three types of output format, one for dense matrices, and two for sparse matrices. We describe them as follows:
- `Dense Matrix Format:` We refer to this format as `DENSE`. DENSE stores one line for each row and the values in each row are separated by white spaces. 
- `Coordinate List Format:` We refer to this format as `COO`. COO stores a list of (row, column, value) tuples. The indices are sorted by row index then column index. The column and row indices are 0-based.
- `List of Lists`:  We refer to this format as `LIL`. LIL stores one list per row, with each entry containing the column index and the value. Entries are comma separated.
 
The user can specify the output Format using the option `-DoutFmt` that will be described later in this document.

Running sPCA-spark in the local mode
=====================================
The next step is to run sPCA-spark on a small toy matrix. There is an example script located in `sPCA/spca-spark/spca-spark_example.sh`. First, you need to set the environment variable `SPARK_HOME` to the directory where Spark is downloaded and installed:
```
export SPARK_HOME=<path/to/spark/directory> (e.g., /usr/lib/spark-1.0.0)
```
You can then run the example through the following command:
```
sPCA/spca-spark/spca-spark_example.sh local
```
where `local` means that the Spark code will run on the local machine. If the examples runs correctly, you should see a message saying `Principal components computed successfully`. The output will be written in `sPCA/spca-spark/output/`.
The example involves a command similar to the following:
```
$SPARK_HOME/bin/spark-submit --class org.qcri.sparkpca.SparkPCA --master <master_url> --driver-java-options "-Di=<path/to/input/matrix> -Do=<path/to/outputfolder> -Drows=<number of rows> -Dcols=<number of columns> -Dpcs=<number of principal components> [-DerrSampleRate=<Error sampling rate>] [-DmaxIter=<max iterations>] [-DoutFmt=<output format>] [-DComputeProjectedMatrix=<0/1 (compute projected matrix or not)>]" target/sparkPCA-1.0.jar 
```
This command runs sPCA on top of Spark in the local machine with one worker thread. The following is a description of the command-line arguments of sPCA:
- `<master-url>: `The master URL for the cluster (e.g. spark://23.195.26.187:7077), it is set to `local[K]` for running Spark in the local mode with *K* threads (ideally, set *K* to the number of cores on your machine). If this argument is set to `local`, the applications runs locally on one worker thread (i.e., no parlellism at all).
-	`<path/to/input/matrix>:` File or directory that contains an input matrix in the SequenceFile Format `<IntWritable key, VectorWritable value>`.
-	`<path/to/outputfolder>:` The directory where the resulting principal components is written
-	`<number of rows>:` Number of rows for the input matrix 
-	`<number of columns>:` Number of columns for the input matrix 
-	`<number of principal components>:` Number of desired principal components 
-	`[<Error sampling rate>](optional):` The error sampling rate [0-1] that is used for computing the error, It can be set to 0.01 to compute the error for only a small sample of the matrix, this speeds up the computations significantly 
- `[<max iterations>] (optional):` Maximum number of iterations before terminating, the default is 3
- `[<output format>] (optional):` One of three supported output formats (DENSE/COO/LIL), the default is DENSE. See Section Output Format for more details.
- `[<0/1 (compute projected matrix or not)>] (optional)` :  0 or 1 value that specifies whether the user wants to project the input matrix on the principal components or not. 1 means that the projected matrix will be computed, and 0 means it will not be computed. The projected matrix is written in the output folder specified  by `-DOutput`
