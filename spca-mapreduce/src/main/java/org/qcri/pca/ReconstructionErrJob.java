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
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
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
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.Closeables;

/**
 * Xc = Yc * Y2X
 * 
 * ReconY = Xc * C'
 * 
 * Err = ReconY - Yc
 * 
 * Norm2(Err) = abs(Err).zSum().max()
 * 
 * To take the sparse matrixes into account we receive the mean separately:
 * 
 * X = (Y - Ym) * Y2X = X - Xm, where X=Y*Y2X and Xm=Ym*Y2X
 * 
 * ReconY = (X - Xm) * C' = X*C' - Xm*C'
 * 
 * Err = X*C' - Xm*C' - (Y - Ym) = X*C' - Y - Zm, where where Zm=Xm*C'-Ym
 * 
 * @author maysam yabandeh
 */
public class ReconstructionErrJob extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(ReconstructionErrJob.class);

  public static final String MATRIXY2X = "matrixY2X";
  public static final String RECONSTRUCTIONMATRIX = "matrixRecon";
  public static final String YCOLS = "yDimension";
  public static final String XCOLS = "xDimension";
  public static final String ZMPATH = "zmPath";
  public static final String YMPATH = "ymPath";
  public static final String ERRSAMPLERATE = "errSampleRate";
  
  /**
   * The norm of the reconstruction error matrix
   */
  public double reconstructionError = -1;
  
  /**
   * The norm of the input matrix
   */
  public double yNorm = -1;
  
  /**
   * The norm of the input matrix after centralization
   */
  public double centralizedYNorm = -1;
  
  /**
   * Refer to {@link ReconstructionErrJob} for explanation of the job. In short:
   * 
   * X = Y * Y2X
   * 
   * Err = (X - Xm) * C' - (Y - Ym)
   * 
   * @param matrixY
   *          the input matrix Y
   * @param matrixY2X
   *          the in-memory matrix to generate X
   * @param matrixC
   *          the in-memory matrix to reconstruct Y
   * @param C_central
   *          the central version of matrixC
   * @param Ym
   *          the mean vector of Y
   * @param Xm
   *          = Ym * matrixY2X
   * @param conf
   *          the configuration
   * @param tmpPath
   *          the temporary path
   * @param id
   *          the unique id to name the files in HDFS
   * @return the norm-2 of the the Err matrix 
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public double reconstructionErr(DistributedRowMatrix matrixY,
      DistributedRowMatrix matrixY2X, DistributedRowMatrix matrixC,
      Matrix C_central, Vector Ym, DenseVector Xm, final float ERR_SAMPLE_RATE, 
      Configuration conf, Path tmpPath, String id)
      throws IOException, InterruptedException, ClassNotFoundException {
    DenseVector Zm = new DenseVector(C_central.numRows());
    PCACommon.vectorTimesMatrixTranspose(Xm, (DenseMatrix) C_central, Zm);
    Zm = (DenseVector) Zm.minus(Ym);
    
    Path resPath = new Path(tmpPath, "reconstructionErr" + id);
    FileSystem fs = FileSystem.get(resPath.toUri(), conf);
    if (!fs.exists(resPath)) {
      Path ZmPath = PCACommon.toDistributedVector(Zm, tmpPath, "Zm" + id, conf);
      Path YmPath = PCACommon.toDistributedVector(Ym, tmpPath, "Ymforerr" + id, conf);
      run(conf,
          matrixY.getRowPath(), matrixY2X.getRowPath(),
          matrixY2X.numRows(), matrixY2X.numCols(), matrixC.getRowPath(),
          ZmPath.toString(), YmPath.toString(), resPath, ERR_SAMPLE_RATE);
    } else {
      log.warn("---------- Skip ReconstructionErrJob - already exists: " + resPath);
    }
    loadResults(resPath, conf);
    
    log.info("0 is reconstruction err, 1 is Y norm (err/norm), "
        + "2 is Y-Ym norm (err/norm)");
    log.info("The error of 0 is " + reconstructionError);
    log.info("The error of 1 is " + yNorm + " (" + reconstructionError / yNorm
        + ")");
    log.info("The error of 2 is " + centralizedYNorm + " ("
        + reconstructionError / centralizedYNorm + ")");
    double error = reconstructionError / centralizedYNorm;
    return error;
  }
  
  /**
   * Refer to {@link ReconstructionErrJob} for explanation of the job
   * 
   * @param conf
   *          the configuration
   * @param yPath
   *          the path to input matrix Y
   * @param y2xPath
   *          the path to in-memory matrix Y2X, where X = Y * Y2X
   * @param yCols
   *          the number of columns in Y
   * @param xCols
   *          the number of columns in X
   * @param cPath
   *          the path to in-memory matrix C, where ReconY = Xc * C'
   * @param zmPath
   *          the path to vector Zm, where Zm = Ym * Y2X * C' - Ym
   * @param ymPath
   *          the path the the mean vector Ym
   * @param outPath
   *          the output path
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public void run(Configuration conf, 
                                Path yPath, 
                                Path y2xPath, 
                                int yCols,
                                int xCols,
                                Path cPath, 
                                String zmPath, 
                                String ymPath, 
                                Path outPath,
                                final float ERR_SAMPLE_RATE) 
                                    throws IOException, InterruptedException, 
                                    ClassNotFoundException {
    conf.set(MATRIXY2X, y2xPath.toString());
    conf.set(RECONSTRUCTIONMATRIX, cPath.toString());
    conf.set(ZMPATH, zmPath);
    conf.set(YMPATH, ymPath);
    conf.setInt(YCOLS, yCols);
    conf.setInt(XCOLS, xCols);
    conf.set(ERRSAMPLERATE, ""+ERR_SAMPLE_RATE);
    FileSystem fs = FileSystem.get(yPath.toUri(), conf);
    yPath = fs.makeQualified(yPath);
    outPath = fs.makeQualified(outPath);
    Job job = new Job(conf);
    FileInputFormat.addInputPath(job, yPath);
    FileOutputFormat.setOutputPath(job, outPath);
    job.setJobName("ReconErrJob-" + yPath.getName());
    job.setJarByClass(ReconstructionErrJob.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setNumReduceTasks(1);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(MyMapper.class);
    job.setReducerClass(MyReducer.class);
    job.setNumReduceTasks(1);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(DoubleWritable.class);
    job.submit();
    job.waitForCompletion(true);
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ReconstructionErrJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    addOption("yCols", "yCols", "Number of cols of the first input matrix", true);
    addOption("xCols", "xCols", "Number of cols of the reconstruction matrix", true);
    addOption("y", "y", "Path to Y, the input matrix", true);
    addOption("d", "d", "Path to D, where Y * D = X", true);
    addOption("c", "c", "Path to C, where X * C' = reconY", true);
    addOption(YMPATH, "ym",
        "The name of the file that contains Ym, the Y mean", true);
    addOption(ZMPATH, "zm",
        "The name of the file that contains Zm mean", true);
    
    Map<String, List<String>> argMap = parseArguments(strings);
    if (argMap == null) {
      return -1;
    }

    String zmFileName = getOption(ZMPATH);
    String meanFileName = getOption(YMPATH);
    int yCols = Integer.parseInt(getOption("yCols"));
    int xCols = Integer.parseInt(getOption("xCols"));
    this.run(new Configuration(), new Path(getOption("y")), new Path(
        getOption("d")), yCols, xCols, new Path(getOption("c")), zmFileName,
        meanFileName, getOutputPath(), 1);
    return 0;
  }

  public void loadResults(Path outDirPath, Configuration conf) throws IOException {
    Path finalNumberFile = new Path(outDirPath, "part-r-00000");
    SequenceFileIterator<IntWritable, DoubleWritable> iterator = 
        new SequenceFileIterator<IntWritable, DoubleWritable>(
        finalNumberFile, true, conf);
    try {
      while (iterator.hasNext()) {
        Pair<IntWritable, DoubleWritable> next = iterator.next();
        readIndividualResult(next.getFirst().get(), next.getSecond().get());
      }
    } finally {
      Closeables.close(iterator, false);
    }
  }
  
  private void readIndividualResult(int key, double value) throws IOException {
    switch (key) {
    case 0:
      reconstructionError = value;
      break;
    case 1:
      yNorm = value;
      break;
    case 2:
      centralizedYNorm = value;
      break;
    default:
      throw new IOException("Unknown key in reading the results: " + key);
    }
  }
  
  public static class MyMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    //input variables
    private DenseMatrix matrixC;
    private DenseMatrix matrixY2X;
    private DenseVector zm;
    private DenseVector ym;
    private float ERR_SAMPLE_RATE; //the sampling rate
    //variables that will be filled by the map method
    private DenseVector xi;
    private DenseVector xiCt;
    private DenseVector sumOfErr;
    private DenseVector sumOfyi;
    private DenseVector sumOfyc;
    DoubleWritable doubleWritable = new DoubleWritable();

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      Path cMemMatrixPath = new Path(conf.get(RECONSTRUCTIONMATRIX));
      Path dMemMatrixPath = new Path(conf.get(MATRIXY2X));
      Path zmPath = new Path(conf.get(ZMPATH));
      Path meanPath = new Path(conf.get(YMPATH));
      int inMemMatrixNumRows = conf.getInt(YCOLS, 0);
      int inMemMatrixNumCols = conf.getInt(XCOLS, 0);
      ERR_SAMPLE_RATE = conf.getFloat(ERRSAMPLERATE, 1);
      Path tmpPath = cMemMatrixPath.getParent();
      DistributedRowMatrix distMatrix = new DistributedRowMatrix(
          cMemMatrixPath, tmpPath, inMemMatrixNumRows, inMemMatrixNumCols);
      distMatrix.setConf(conf);
      matrixC = PCACommon.toDenseMatrix(distMatrix);
      distMatrix = new DistributedRowMatrix(
          dMemMatrixPath, tmpPath, inMemMatrixNumRows, inMemMatrixNumCols);
      distMatrix.setConf(conf);
      matrixY2X = PCACommon.toDenseMatrix(distMatrix);
      try {
        zm = PCACommon.toDenseVector(zmPath, conf);
        ym = PCACommon.toDenseVector(meanPath, conf);
      } catch (IOException e) {
        e.printStackTrace();
      }
      xiCt = new DenseVector(matrixC.numRows());
      sumOfErr = new DenseVector(matrixC.numRows());
      sumOfyi = new DenseVector(matrixC.numRows());
      sumOfyc = new DenseVector(matrixC.numRows());
    }

    @Override
    public void map(IntWritable iw, VectorWritable vw, Context context)
        throws IOException {
      if (PCACommon.pass(ERR_SAMPLE_RATE))
        return;

      Vector yi = vw.get();
      if (xi == null)
        xi = new DenseVector(matrixY2X.numCols());
      PCACommon.sparseVectorTimesMatrix(yi, matrixY2X, xi);

      PCACommon.vectorTimesMatrixTranspose(xi, matrixC, xiCt);
      denseVectorSubtractSparseSubtractDense(xiCt, yi, zm);
      sumOfErr.assign(xiCt, new DoubleDoubleFunction() {
        @Override
        public double apply(double arg1, double arg2) {
          return arg1 + Math.abs(arg2);
        }
      });
      denseVectorPlusAbsSparseVector(sumOfyi, yi);
      denseVectorPlusAbsDenseDiff(sumOfyc, yi, ym);
    }
    
    @Override
    public void cleanup(Context context) throws InterruptedException,
        IOException {
      context.write(new IntWritable(0), new VectorWritable(sumOfErr));
      context.write(new IntWritable(1), new VectorWritable(sumOfyi));
      context.write(new IntWritable(2), new VectorWritable(sumOfyc));
    }    
  }

  public static class MyReducer extends
      Reducer<IntWritable, VectorWritable, IntWritable, DoubleWritable> {
    @Override
    public void reduce(IntWritable id,
                       Iterable<VectorWritable> sums,
                       Context context) throws IOException, InterruptedException {
      Iterator<VectorWritable> it = sums.iterator();
      if (!it.hasNext()) {
        return;
      }
      DenseVector sumVector = null;
      while (it.hasNext()) {
        Vector vec = it.next().get();
        if (sumVector == null) {
          sumVector = new DenseVector(vec.size());
        }
        sumVector.assign(vec, Functions.PLUS); 
      }
      double max = sumVector.maxValue();
      context.write(id, new DoubleWritable(max));
    }
  }
  
  //utility functions
  static void denseVectorPlusAbsDenseDiff(
      DenseVector denseVector, Vector sparseVector, DenseVector meanVector) {
    for (int i = 0; i < denseVector.size(); i++) {
      double denseV = denseVector.getQuick(i);
      double v = sparseVector.getQuick(i);
      double mean = meanVector.getQuick(i);
      denseVector.setQuick(i, denseV + Math.abs(v-mean));
    }
  }

  static void denseVectorPlusAbsSparseVector(
      DenseVector denseVector, Vector sparseVector) {
    Iterator<Vector.Element> nonZeroElements = sparseVector.nonZeroes().iterator();
    while (nonZeroElements.hasNext()) {
      Vector.Element e = nonZeroElements.next();
      int index = e.index();
      double v = e.get();
      double prevV = denseVector.getQuick(index);
      denseVector.setQuick(index, prevV + Math.abs(v));
    }
  }

  static void denseVectorSubtractSparseSubtractDense(DenseVector mainVector,
      Vector subtractor1, DenseVector subtractor2) {
    int nCols = mainVector.size();
    for (int c = 0; c < nCols; c++) {
      double v = mainVector.getQuick(c);
      v -= subtractor1.getQuick(c);
      v -= subtractor2.getQuick(c);
      mainVector.setQuick(c, v);
    }
  }

}
