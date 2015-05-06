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
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.Closeables;

/**
 * This job calculates the mean and span of the columns of a
 * {@link DistributedRowMatrix}
 * 
 * @author maysam yabandeh
 */
public class MeanAndSpanJob {
  private static final Logger log = LoggerFactory.getLogger(MeanAndSpanJob.class);
  public static final int MEANVECTOR = 0;
  public static final int MINVECTOR = 1;
  public static final int MAXVECTOR = 2;
  public static final int SPANVECTOR = 3;
  public static final IntWritable meanIntWritable = new IntWritable(MEANVECTOR);
  public static final IntWritable minIntWritable = new IntWritable(MINVECTOR);
  public static final IntWritable maxIntWritable = new IntWritable(MAXVECTOR);
  public static final IntWritable spanIntWritable = new IntWritable(SPANVECTOR);
  public static final boolean NORMALIZE_MEAN = true;

  private Vector meanVector = null;

  public Vector getMeanVector() {
    return meanVector;
  }

  private Vector spanVector = null;

  public Vector getSpanVector() {
    return spanVector;
  }

  public Path getMeanSpanPath(Path outputDir) {
    Path meanSpanPath = new Path(outputDir, "part-r-00000");
    return meanSpanPath;
  }

  /**
   * Computes the column-wise mean and span of a DistributedRowMatrix
   * @throws IOException 
   * 
   */
  public Path compuateMeanAndSpan(Path inputPath, Path outputPath,
      DenseVector resMean, boolean normalizeMean, Configuration conf, String id)
      throws IOException {
    Path meanSpanDirPath = new Path(outputPath, "meanAndSpan" + id);
    FileSystem fs = FileSystem.get(inputPath.toUri(), conf);
    meanSpanDirPath = fs.makeQualified(meanSpanDirPath);
    if (!fs.exists(meanSpanDirPath)) {
      Path rowPath = fs.makeQualified(inputPath);
      run(conf, rowPath, meanSpanDirPath);
    } else {
      log.warn("--------- Skip MeanAndSpanJob - already exists" + meanSpanDirPath);
    }
    Path meanSpanPath = getMeanSpanPath(meanSpanDirPath);
    loadResults(meanSpanPath, normalizeMean, conf);
    resMean.assign(getMeanVector());
    return meanSpanPath;
  }

  /**
   * Job for calculating column-wise mean and span of a
   * {@link DistributedRowMatrix}
   * 
   * @param initialConf
   *          the initial configuration
   * @param inputPath
   *          the path to the matrix
   * @param outputVectorTmpPath
   *          the path to which the result vectors will be written
   * @throws IOException
   *           in case of any error
   */
  public void run(Configuration initialConf, Path inputPath,
      Path outputVectorTmpPath) throws IOException {
    try {
      Job job = new Job(initialConf);
      job.setJobName("MeanAndSpan");
      job.setJarByClass(MeanAndSpanJob.class);
      FileOutputFormat.setOutputPath(job, outputVectorTmpPath);
      FileInputFormat.addInputPath(job, inputPath);
      job.setInputFormatClass(SequenceFileInputFormat.class);
      job.setOutputFormatClass(SequenceFileOutputFormat.class);

      job.setMapperClass(MeanAndSpanMapper.class);
      job.setReducerClass(MeanAndSpanReducer.class);
      job.setNumReduceTasks(1);// it has to be one
      job.setMapOutputKeyClass(IntWritable.class);
      job.setMapOutputValueClass(VectorWritable.class);
      job.setOutputKeyClass(IntWritable.class);
      job.setOutputValueClass(VectorWritable.class);
      job.submit();
      job.waitForCompletion(true);
    } catch (Throwable thr) {
      thr.printStackTrace();
      if (thr instanceof IOException)
        throw (IOException) thr;
      else
        throw new IOException(thr);
    }
  }

  /**
   * After running the job, we can load the results from HDFS with this method
   * 
   * @param meanSpanPath
   *          the path to the single file having the results
   * @param normalizeMean
   *          normalize the mean as well
   * @param conf
   *          the configuration
   * @throws IOException
   *           when face problem parsing the result file
   */
  public void loadResults(Path meanSpanPath, boolean normalizeMean, Configuration conf)
      throws IOException {
    SequenceFileIterator<IntWritable, VectorWritable> iterator = 
        new SequenceFileIterator<IntWritable, VectorWritable>(
        meanSpanPath, true, conf);
    try {
      Pair<IntWritable, VectorWritable> next;
      next = iterator.next();
      if (next.getFirst().get() == MEANVECTOR)
        meanVector = next.getSecond().get();
      else
        spanVector = next.getSecond().get();
      next = iterator.next();
      if (next.getFirst().get() == MEANVECTOR)
        meanVector = next.getSecond().get();
      else
        spanVector = next.getSecond().get();
    } finally {
      Closeables.close(iterator, false);
    }
    if (normalizeMean)
      meanVector.assign(spanVector, new DoubleDoubleFunction() {
        @Override
        public double apply(double v, double span) {
          return v / (span != 0 ? span : 1);
        }
      });
  }

  /**
   * Mapper for calculation of column-wise mean.
   */
  public static class MeanAndSpanMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    private Vector runningSum = null;
    private Vector runningMin = null;
    private Vector runningMax = null;
    private Vector runningCnt = null;// cnt of non-zero columns

    /**
     * The mapper computes a running sum, min, max, and cnt of the vectors the
     * task has seen. Element 0 of the running sum vector contains a count of
     * the number of vectors that have been seen. The remaining of its elements
     * contain the column-wise running sum. Nothing is written at this stage
     */
    @Override
    public void map(IntWritable r, VectorWritable v, Context context)
        throws IOException {
      if (runningSum == null) { // first round
        // DenseVector is more efficient since it is used for aggregation
        runningSum = new DenseVector(v.get().size() + 1);// first element is the
                                                         // total
        runningMin = new DenseVector(v.get().size());
        runningMax = runningMin.like();
        runningCnt = runningMin.like();
        // initialize the the current record
        runningSum.set(0, 1);
        runningSum.viewPart(1, v.get().size()).assign(v.get());
        runningMin.assign(v.get());
        runningMax.assign(v.get());
        vectorAssign(runningCnt, v.get(), CNT);
      } else {
        runningSum.set(0, runningSum.get(0) + 1);
        // The NaN values are considered zero
        vectorAssign(runningSum.viewPart(1, v.get().size()), v.get(), PLUSNaN);
        // Min and Max on sparse vectors skip over zero elements
        // This is later taken care of by keeping the count in {@link
        // #runningCnt}
        vectorAssign(runningMin, v.get(), MINNaN);
        vectorAssign(runningMax, v.get(), MAXNaN);
        vectorAssign(runningCnt, v.get(), CNT);
      }
    }

    /**
     * The column-wise aggregate vectors are written at the cleanup stage. A
     * single reducer is forced so null can be used for the key
     */
    @Override
    public void cleanup(Context context) throws InterruptedException,
        IOException {
      if (runningSum != null) {
        updateMinMax(runningMin, runningMax, runningCnt,
            (int) runningSum.getQuick(0));
        context.write(meanIntWritable, new VectorWritable(runningSum));
        context.write(minIntWritable, new VectorWritable(runningMin));
        context.write(maxIntWritable, new VectorWritable(runningMax));
      }
    }

    /**
     * Man and Max over sparse vector skip over zero elements. We take the zero
     * elements into account in this method.
     * 
     * @param minV
     *          the min
     * @param maxV
     *          the max
     * @param cntV
     *          the number of non-zero elements
     * @param total
     *          number of processed rows
     */
    private void updateMinMax(Vector minV, Vector maxV, Vector cntV, int total) {
      for (int i = 0; i < cntV.size(); i++) {
        int cnt = (int) cntV.getQuick(i);
        if (cnt != total) {
          // this implies that there was a zero element not counted in
          // computing min and max. So, we count it in now.
          double min = minV.getQuick(i);
          double max = maxV.getQuick(i);
          min = Math.min(min, 0);
          max = Math.max(max, 0);
          minV.setQuick(i, min);
          maxV.setQuick(i, max);
        }
      }
    }
  }

  /**
   * The reducer adds the partial column-wise sums from each of the mappers to
   * compute the total column-wise sum. The total sum is then divided by the
   * total count of vectors to determine the column-wise mean.
   * 
   * This also computes min and max for computing span
   */
  public static class MeanAndSpanReducer extends
      Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    Vector meanVector = null;
    Vector minVector = null;
    Vector maxVector = null;

    @Override
    public void reduce(IntWritable n, Iterable<VectorWritable> vectors,
        Context context) throws IOException, InterruptedException {
      // Aggregate partial column-wise data from mappers
      for (VectorWritable v : vectors) {
        int key = n.get();
        switch (key) {
        case MEANVECTOR:
          if (meanVector == null) {
            meanVector = new DenseVector(v.get().size());
            meanVector.assign(v.get());
          } else
            vectorAssign(meanVector,v.get(), PLUSNaN);
          break;
        case MINVECTOR:
          if (minVector == null) {
            minVector = new DenseVector(v.get().size());
            minVector.assign(v.get());
          } else
            vectorAssign(minVector,v.get(), MINNaN);
          break;
        case MAXVECTOR:
          if (maxVector == null) {
            maxVector = new DenseVector(v.get().size());
            maxVector.assign(v.get());
          } else
            vectorAssign(maxVector,v.get(), MAXNaN);
          break;
        default:
          throw new IOException("Unknown key in the reducer: " + key);
        }
      }
    }

    @Override
    public void cleanup(Context context) throws InterruptedException,
        IOException {
      if (meanVector == null || minVector == null || maxVector == null)
        throw new IOException(
            "The reduce phase did not observe all the data expected, "
                + "one of mean, min, and max is missing!");
      // Divide total column-wise sum by count of vectors, which corresponds to
      // the number of rows in the DistributedRowMatrix
      VectorWritable outputVectorWritable = new VectorWritable();
      outputVectorWritable.set(meanVector.viewPart(1, meanVector.size() - 1)
          .divide(meanVector.get(0)));
      context.write(meanIntWritable, outputVectorWritable);
      outputVectorWritable = new VectorWritable();
      outputVectorWritable.set(maxVector.minus(minVector));
      context.write(spanIntWritable, outputVectorWritable);
    }
  }
  
  //utility functions
  /** a + b, accepts NaN as input */
  static class PLUSNaNClass implements ZeroIndifferentFunc {
    @Override
    public double apply(double a, double b) {
      if (Double.isNaN(a))
        a = 0;
      if (Double.isNaN(b))
        b = 0;
      return a + b;
    }
  };

  static final PLUSNaNClass PLUSNaN = new PLUSNaNClass();

  /** Math.max(a,b), accepts NaN as input */
  static class MAXNaNClass implements ZeroIndifferentFunc {
    @Override
    public double apply(double a, double b) {
      if (Double.isNaN(a))
        a = 0;
      if (Double.isNaN(b))
        b = 0;
      return Math.max(a, b);
    }
  };

  static final MAXNaNClass MAXNaN = new MAXNaNClass();

  /** Math.min(a,b), accepts NaN as input */
  static class MINNaNClass implements ZeroIndifferentFunc {
    @Override
    public double apply(double a, double b) {
      if (Double.isNaN(a))
        a = 0;
      if (Double.isNaN(b))
        b = 0;
      return Math.min(a, b);
    }
  };

  static final MINNaNClass MINNaN = new MINNaNClass();

  /** count the number of non-zero elements */
  static class CNTClass implements ZeroIndifferentFunc {
    @Override
    public double apply(double a, double b) {
      return a+1;
    }
  };

  static final CNTClass CNT = new CNTClass();
  

  /**
   * Meaning that zero elements would not change the state and can be skipped
   */
  public interface ZeroIndifferentFunc {
    /**
     * Apply the function to the arguments and return the result
     *
     * @param arg1 a double for the first argument
     * @param arg2 a double for the second argument
     * @return the result of applying the function
     */
    double apply(double arg1, double arg2);
  }

  /**
   * This method overrides the Vector.assign method to allow optimization for
   * ZeroIndifferent functions
   * 
   * @param vector
   *          the vector to be updated
   * @param other
   *          the other vector
   * @param function
   *          the function that operates on elements of the two vectors
   * @return the modified vector
   */
  static public Vector vectorAssign(Vector vector, Vector other, ZeroIndifferentFunc function) {
    if (vector.size() != other.size()) {
      throw new CardinalityException(vector.size(), other.size());
    }
    // special case: iterate only over the non-zero elements of the vector to
    // add
    Iterator<Element> it = other.nonZeroes().iterator();
    Element e;
    while (it.hasNext() && (e = it.next()) != null) {
      double val = vector.getQuick(e.index());
      double newVal = function.apply(val, e.get());
      vector.setQuick(e.index(), newVal);
    }
    return vector;
  }

}
