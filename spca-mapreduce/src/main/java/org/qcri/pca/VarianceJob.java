/**
 * QCRI, sPCA LICENSE
 * sPCA is a scalable implementation of Principal Component Analysis (PCA) on of Spark and MapReduce
 *
 * Copyright (c) 2015, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
*/

package org.qcri.pca;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.Closeables;

/**
 * Computes part of variance that requires a distributed job
 * 
 * xcty = Sum (xi * C' * yi')
 * 
 * We also regenerate xi on demand by the following formula:
 * 
 * xi = yi * y2x
 * 
 * To make it efficient for uncentralized sparse inputs, we receive the mean
 * separately:
 * 
 * xi = (yi - ym) * y2x = yi * y2x - xm, where xm = ym*y2x
 * 
 * xi * C' * (yi-ym)' = xi * ((yi-ym)*C)' = xi * (yi*C - ym*C)'
 * 
 * @author maysam yabandeh
 */
public class VarianceJob extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(VarianceJob.class);

  public static final String MATRIXY2X = "matrixY2X";
  public static final String MATRIXC = "matrixC";
  public static final String XMPATH = "xm.path";
  public static final String YMPATH = "ym.path";

  @Override
  public int run(String[] arg0) throws Exception {
    throw new Exception("Unimplemented");
  }

  /**
   * refer to {@link VarianceJob} for job description. In short, it does: for i
   * in 1:N: sum += (xi-xm) * C' * (yi-ym)'
   * 
   * @param matrixY
   *          the input matrix Y
   * @param ym
   *          the column mean of Y
   * @param matrixY2X
   *          the matrix to generate X
   * @param xm
   *          = ym * Y2X
   * @param matrixC
   *          the matrix of principal components
   * @param tmpPath
   *          the temporary path in HDFS
   * @param conf
   *          the configuration
   * @param id
   *          the unique id to name files in HDFS
   * @return
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public double computeVariance(DistributedRowMatrix matrixY, Vector ym,
      DistributedRowMatrix matrixY2X, Vector xm, DistributedRowMatrix matrixC,
      Path tmpPath, Configuration conf, String id) throws IOException,
      InterruptedException, ClassNotFoundException {
    Path xmPath = PCACommon.toDistributedVector(xm, tmpPath, "Xm-varianceJob"
        + id, conf);
    Path ymPath = PCACommon.toDistributedVector(ym, tmpPath, "Ym-varianceJob"
        + id, conf);

    Path outPath = new Path(tmpPath, "Variance"+id);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    if (!fs.exists(outPath)) {
      run(conf, matrixY.getRowPath(), ymPath.toString(), matrixY2X.getRowPath()
          .toString(), xmPath.toString(), matrixC.getRowPath().toString(),
          outPath);
    } else {
      log.warn("---------- Skip variance - already exists: " + outPath);
    }
    loadResult(outPath, conf);
    return finalSum;// finalNumber;
  }
  
  public void run(Configuration conf, Path yPath, String ymPath,
      String matrixY2XDir, String xmPath, String matrixCDir, Path outPath)
      throws IOException, InterruptedException, ClassNotFoundException {
    conf.set(MATRIXY2X, matrixY2XDir);
    conf.set(MATRIXC, matrixCDir);
    conf.set(XMPATH, xmPath);
    conf.set(YMPATH, ymPath);
    FileSystem fs = FileSystem.get(yPath.toUri(), conf);
    yPath = fs.makeQualified(yPath);
    outPath = fs.makeQualified(outPath);
    Job job = new Job(conf);
    FileInputFormat.addInputPath(job, yPath);
    FileOutputFormat.setOutputPath(job, outPath);
    job.setJobName("VarianceJob-" + yPath.getName());
    job.setJarByClass(VarianceJob.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setNumReduceTasks(1);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(MyMapper.class);
    job.setReducerClass(MyReducer.class);
    job.setMapOutputKeyClass(NullWritable.class);
    job.setMapOutputValueClass(DoubleWritable.class);
    job.setOutputKeyClass(NullWritable.class);
    job.setOutputValueClass(DoubleWritable.class);
    job.submit();
    job.waitForCompletion(true);
  }

  /**
   * The final results after running the job
   */
  public double finalSum = -1;
  public void loadResult(Path outDirPath, Configuration conf)
      throws IOException {
    Path finalNumberFile = new Path(outDirPath, "part-r-00000");
    SequenceFileIterator<NullWritable, DoubleWritable> iterator = new SequenceFileIterator<NullWritable, DoubleWritable>(
        finalNumberFile, true, conf);
    try {
      Pair<NullWritable, DoubleWritable> next = iterator.next();
      finalSum = next.getSecond().get();
    } finally {
      Closeables.close(iterator, false);
    }
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new VarianceJob(), args);
  }

  public static class MyMapper extends
      Mapper<IntWritable, VectorWritable, NullWritable, DoubleWritable> {
    private DenseVector xm;
    private DenseVector ym;
    private DenseVector ymC; // ym * C
    private DenseVector xi;
    private DenseVector yiC; // yi * C
    private DenseMatrix matrixY2X;
    private DenseMatrix matrixC;
    private double sum = 0;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      Path y2xMatrixPath = new Path(conf.get(MATRIXY2X));
      Path cMatrixPath = new Path(conf.get(MATRIXC));
      Path xmPath = new Path(conf.get(XMPATH));
      Path ymPath = new Path(conf.get(YMPATH));
      try {
        xm = PCACommon.toDenseVector(xmPath, conf);
        ym = PCACommon.toDenseVector(ymPath, conf);
      } catch (IOException e) {
        e.printStackTrace();
      }
      Path tmpPath = cMatrixPath.getParent();
      DistributedRowMatrix distMatrix = new DistributedRowMatrix(y2xMatrixPath,
          tmpPath, ym.size(), xm.size());
      distMatrix.setConf(conf);
      matrixY2X = PCACommon.toDenseMatrix(distMatrix);
      distMatrix = new DistributedRowMatrix(cMatrixPath, tmpPath, ym.size(),
          xm.size());
      distMatrix.setConf(conf);
      matrixC = PCACommon.toDenseMatrix(distMatrix);
      ymC = new DenseVector(matrixC.numCols());
      PCACommon.denseVectorTimesMatrix(ym, matrixC, ymC);
    }

    @Override
    public void map(IntWritable iw, VectorWritable vw, Context context)
        throws IOException {
      Vector yi = vw.get();
      // regenerate xi
      if (xi == null)
        xi = new DenseVector(xm.size());
      PCACommon.sparseVectorTimesMatrix(yi, matrixY2X, xi);
      /**
       * xi = xi - xm
       * 
       * sum += xi * (yi*C - ym*C)'
       */
      xi.assign(xm, Functions.MINUS);
      if (yiC == null)
        yiC = new DenseVector(xm.size());
      PCACommon.sparseVectorTimesMatrix(yi, matrixC, yiC);
      yiC.assign(ymC, Functions.MINUS);
      double dotRes = xi.dot(yiC);
      sum += dotRes;
    }

    @Override
    public void cleanup(Context context) throws InterruptedException,
        IOException {
      context.write(NullWritable.get(), new DoubleWritable(sum));
    }
  }

  public static class MyReducer extends
      Reducer<NullWritable, DoubleWritable, NullWritable, DoubleWritable> {
    @Override
    public void reduce(NullWritable key, Iterable<DoubleWritable> sums,
        Context context) throws IOException, InterruptedException {
      Iterator<DoubleWritable> it = sums.iterator();
      if (!it.hasNext()) {
        return;
      }
      double sum = 0;
      while (it.hasNext()) {
        double v = it.next().get();
        sum += v;
      }
      context.write(key, new DoubleWritable(sum));
    }
  }

}
