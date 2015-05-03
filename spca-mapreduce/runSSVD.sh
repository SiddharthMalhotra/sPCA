export HADOOP_CLASSPATH=/home/myabandeh/mahout0.8/mahout-core-0.8-SNAPSHOT-job.jar:/home/myabandeh/mahout0.8/mahout-core-0.8-SNAPSHOT.jar:/home/myabandeh/mahout0.8/mahout-math-0.8-SNAPSHOT.jar
export LIBJARS=/home/myabandeh/mahout0.8/mahout-core-0.8-SNAPSHOT-job.jar,/home/myabandeh/mahout0.8/mahout-core-0.8-SNAPSHOT-job.jar,/home/myabandeh/mahout0.7/mahout-math-0.7.jar,/home/myabandeh/mahout0.8/commons-cli-2.0-mahout.jar
export PATH=$PATH:/usr/lib/hadoop:/usr/bin/hadoop:/home/myabandeh/mahout0.7/mahout-math-0.7.jar:/home/myabandeh/mahout0.8/mahout-core-0.8-SNAPSHOT-job.jar:/home/myabandeh/mahout0.8/mahout-core-0.8-SNAPSHOT.jar:/home/myabandeh/#mahout0.8/commons-cli-2.0-mahout.jar

#hadoop jar mahout-core-0.8-SNAPSHOT-job.jar org.apache.mahout.math.hadoop.stochasticsvd.SSVDCli -i 1Ginput -o ssvdout -q 0 -p 15 -t 1 -k 50 -ow -pca true --tempDir ssvdtmp
#hadoop jar mahout-core-0.8-SNAPSHOT-job.jar org.apache.mahout.math.hadoop.stochasticsvd.SSVDCli -i ssvderrout/partitionedY -o test/ssvdout -q 0 -p 15 -t 1 -k 50 -ow -pca true --tempDir test/ssvdtmp

for i in 0
do

hadoop jar mahout-core-0.8-SNAPSHOT-job.jar org.apache.mahout.math.hadoop.stochasticsvd.SSVDCli -i tweetinputNormalized -o ssvdtweetout56r/$i -q $i -p 15 -t 56 -k 50 -ow -pca true --tempDir ssvdtweettmp56r/$i 2>&1 | tee ssvdtwitt56r$i.log

done

