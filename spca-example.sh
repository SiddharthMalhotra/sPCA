# Before running this script you should set the enviroment variable SPARK_HOME (e.g. export SPARK_HOME=/path/to/spark_home_dirrectory)
# Usage:
# ./bin/spark-submit \
# --class org.qcri.sparkpca.SparkPCA
#  --master <master-url> \
#  ... # other options
#  ./target/sparkPCA-1.0.jar \
#  -DInput=<path/to/input/matrix> -DOutput=<path/to/outputfile> -DRows=<number of rows> -DCols=<number of columns> -DPCs=<number of principal components> [-DErr=<error sampling rate>] [-DMaxIter=<max iterations>] [-DComputeProjectedMatrix=<1/0>] [-DOutFmt=<output format>] 



# Description of arguments:
#       <master-url>: The master URL for the cluster (e.g. spark://23.195.26.187:7077), it is set to local for running locally local mode
#       <path/to/input/matrix>: directory that contains an example input matrix in the sequenceFileFormat <IntWritable key, VectorWritable value>.
#       <path/to/outputfile>: The file where the resulting principal components is written
#       <number of rows>: Number of rows for the input matrix
#       <number of columns>: Number of columns for the input matrix : 5
#       <number of principal components>: Number of desired principal components
#       [<Error sampling rate>](optional): The error sampling rate [0-1] that is used for computing the error, It can be set to 0.01 to compute the error for only a small sample of the matrix, this speeds up the computations significantly)
#       [<max iterations>] (optional): Maximum number of iterations before terminating, the default is 3

if [  $# -le 1 ] 
	then 
		echo -e "\nUsage:\n$0 <master_url> \n" 
		exit 1
	fi 
 
master_url=$1  #master url has two options (local, spark://<IP>:7077) 
SCRIPT=$(readlink -f $0) # Absolute path to this script.
SCRIPTPATH=`dirname $SCRIPT` # Absolute path this script is in. /home/user/bin
$SPARK_HOME/bin/spark-submit --class org.qcri.sparkpca.SparkPCA --master $master_url $SCRIPTPATH/target/sparkPCA-1.0.jar -DInput=$SCRIPTPATH/input/seqfiles -DOutput=$SCRIPTPATH/output.txt -DRows=7 -DCols=5 -DErr=1 -DMaxIter=3
