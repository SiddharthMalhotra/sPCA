# Before running this script you should set the enviroment variable SPARK_HOME (e.g. export SPARK_HOME=/path/to/spark_home_dirrectory)
# Usage:
# --class org.qcri.sparkpca.SparkPCA
#  --master <master-url> \
#  ... # other options
#  ./target/sparkPCA-1.0.jar \
#  <path/to/input/matrix> <path/to/outputfile> <number of rows> <number of columns> <number of principal components> [<Error sampling rate>] [<max iterations>]
# ./bin/spark-submit \
	

# Description of arguments:
# 	<master-url>: The master URL for the cluster (e.g. spark://23.195.26.187:7077), it is set to local for running locally local mode 
#	<path/to/input/matrix>: directory that contains an example input matrix in the sequenceFileFormat <IntWritable key, VectorWritable value>.
#	<path/to/outputfile>: The file where the resulting principal components is written
#	<number of rows>: Number of rows for the input matrix 
#	<number of columns>: Number of columns for the input matrix : 5 
#	<number of principal components>: Number of desired principal components 
#	[<Error sampling rate>](optional): The error sampling rate [0-1] that is used for computing the error, It can be set to 0.01 to compute the error for only a small sample of the matrix, this speeds up the computations significantly) 
# 	[<max iterations>] (optional): Maximum number of iterations before terminating, the default is 3

$SPARK_HOME/bin/spark-submit --class org.qcri.sparkpca.SparkPCA --master local target/sparkPCA-1.0.jar seqfiles output.txt 7 5 3 1 3
