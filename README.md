ScalablePCA
===========

Scalable PCA (sPCA) is a scalable implementation of Principal component analysis (PCA) algorithm on top of Spark. sPCA has been tested on Apache Spark 1.0.0 and Linux OS. It should work with more recent Spark versions with only minor modifications; however, switching to another platform (e.g., Mac) will require recompiling the jars. In the following, we will take you through running PCA on a toy matrix. First, we will use Spark local mode. Running in local model does not require setting up a cluster. Next, we will run sPCA on an Amazon EC2 cluster.


Download and Install Spark
==========================

Download  Spark 1.0.0+ [here](https://spark.apache.org/downloads.html). After Spark is downloaded, build it using the following command:

```
$SPARK_HOME/sbt/sbt assembly
}
```

You can also build Spark using Maven by following [this tutorial] (http://spark.apache.org/docs/1.0.0/building-with-maven.html).
Verify that Spark is running by executing the SparkPi example. In the shell, run the following command:
<code snippet>

After the above Spark local program finishes, you should see the computed value of pi (something that's reasonably closer to 3.14).

Clone the ScalablePCA repo
==========================
Open the shell and clone the ScalablePCA github repo:

git clone git://github.com/lintool/Cloud9.git

In order to build sPCA source code, you need to install maven. You can download and install maven by folliwng this [quick tutorial] (http://maven.apache.org/guides/getting-started/maven-in-five-minutes.html). To verify that maven is installed, run the following command:

mvn --version

It should print out your installed version of Maven. After that, you can build sPCA by typing:

mvn package

Put here Build Successful

this command will build the code and you will find a .jar file generated under ScalablePCA/target/SparkPCA.jar


Running ScalablePCA in the local mode
=====================================


Running ScalablePCA on amazon ec2 cluster
=========================================


