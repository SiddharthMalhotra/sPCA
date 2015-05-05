/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.qcri.pca;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.QRDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.VectorFunction;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The driver provides different implementation of PPCA: Probabilistic Principal
 * Component Analysis.
 * 
 * 1. sPCA: PPCA on top of MapReduce
 * 
 * 2. PPCA: sequential PPCA based on the paper from Tipping and Bishop
 * 
 * 3. PPCA_jakob: sequential PPCA based on the <a
 * href="http://lear.inrialpes.fr/~verbeek/software.php">matlab
 * implementation</a> provided by Jakob Verbeek.
 * 
 * @author maysam yabandeh
 * 
 */
public class SPCADriver extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(SPCADriver.class);

  private static final String ROWSOPTION = "N";
  private static final String COLSOPTION = "D";
  private static final String PRINCIPALSOPTION = "d";
  private static final String SPLITFACTOROPTION = "sf";
  private static final String ERRSAMPLE = "errSampleRate";
  private static final String MAXITER = "maxIter";
  private static final String NORMALIZEOPTION = "normalize";
  private static final boolean CALCULATE_ERR_ATTHEEND = false;

  /**
   * The sampling rate that is used for computing the reconstruction error
   */

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.methodOption().create());
    addOption(ROWSOPTION, "rows", "Number of rows");
    addOption(COLSOPTION, "cols", "Number of cols");
    addOption(PRINCIPALSOPTION, "pcs", "Number of principal components");
    addOption(SPLITFACTOROPTION, "sf", "Split each block to increase paralelism");
    addOption(ERRSAMPLE, "errSampleRate",
        "Sampling rate for computing the error (0-1]");
    addOption(MAXITER, "maxIter",
            "Maximum number of iterations before terminating, the default is 3");
    addOption(NORMALIZEOPTION, "normalize",
            "Choose whether you want the input matrix to be  normalized or not, 1 means normalize, 0 means don't normalize");
    if (parseArguments(args) == null) {
      return -1;
    }
    Path input = getInputPath();
    Path output = getOutputPath();
    final int nRows = Integer.parseInt(getOption(ROWSOPTION));
    final int nCols = Integer.parseInt(getOption(COLSOPTION));
    final int nPCs = Integer.parseInt(getOption(PRINCIPALSOPTION));
    final int splitFactor;
    final int normalize;
    final int maxIterations;
    final float errSampleRate;
    if(hasOption(SPLITFACTOROPTION))
    	splitFactor= Integer.parseInt(getOption(SPLITFACTOROPTION, "1"));
    else
    	splitFactor=1;
    if (hasOption(ERRSAMPLE))
    	errSampleRate = Float.parseFloat(getOption(ERRSAMPLE));
    else 
    {
    	 int length = String.valueOf(nRows).length();
    	 if(length <= 4)
    		 errSampleRate= 1;
         else
        	 errSampleRate=(float) (1/Math.pow(10, length-4));
    	 log.warn("error sampling rate set to:  errRate=" + errSampleRate);
    }
    	
    if (hasOption(MAXITER))
       maxIterations = Integer.parseInt(getOption(MAXITER));
    else
    	maxIterations=3;
    if (hasOption(NORMALIZEOPTION))
       normalize = Integer.parseInt(getOption(NORMALIZEOPTION));
    else
    	normalize=0;
    
    Configuration conf = getConf();
    if (conf == null) {
      throw new IOException("No Hadoop configuration present");
    }
    boolean runSequential = getOption(DefaultOptionCreator.METHOD_OPTION)
        .equalsIgnoreCase(DefaultOptionCreator.SEQUENTIAL_METHOD);
    run(conf, input, output, nRows, nCols, nPCs, splitFactor, errSampleRate, maxIterations, normalize, runSequential);
    return 0;
  }

  public void run(Configuration conf, Path input, Path output, final int nRows,
      final int nCols, final int nPCs, final int splitFactor, final float errSampleRate, final int maxIterations, final int normalize, final boolean runSequential)
      throws Exception {
    System.gc();
    if (runSequential)
      runSequential(conf, input, output, nRows, nCols, nPCs);
    else
      runMapReduce(conf, input, output, nRows, nCols, nPCs, splitFactor, errSampleRate, maxIterations, normalize);
  }

  /**
   * Run sPCA
   * 
   * @param conf
   *          the configuration
   * @param input
   *          the path to the input matrix Y
   * @param output
   *          the path to the output (currently for normalization output)
   * @param nRows
   *          number of rows in input matrix
   * @param nCols
   *          number of columns in input matrix
   * @param nPCs
   *          number of desired principal components
   * @param splitFactor
   *          divide the block size by this number to increase parallelism
   * @return the error
   * @throws Exception
   */
  double runMapReduce(Configuration conf, Path input, Path output, final int nRows,
      final int nCols, final int nPCs, final int splitFactor, final float errSampleRate, final int maxIterations, final int normalize) throws Exception {
    Matrix centC = PCACommon.randomMatrix(nCols, nPCs);
    double ss = PCACommon.randSS();
    InitialValues initVal = new InitialValues(centC, ss);
    DistributedRowMatrix distY = new DistributedRowMatrix(input,
        getTempPath(), nRows, nCols);
    distY.setConf(conf);
    /**
     * Here we can control the number of iterations as well as the input size.
     * Can be used to improve initVal by first running on a sample, e.g.:
     * runMapReduce(conf, distY, initVal, ..., 1, 10, 0.001);
     * runMapReduce(conf, distY, initVal, ..., 11, 13, 0.01);
     * runMapReduce(conf, distY, initVal, ..., 14, 1, 1);
     */
    double error = runMapReduce(conf, distY, initVal, output, nRows, nCols, nPCs,
        splitFactor, errSampleRate, maxIterations, normalize);
    return error;
  }

  /**
   * Run sPCA
   * 
   * @param conf
   *          the configuration
   * @param input
   *          the path to the input matrix Y
   * @param output
   *          the path to the output (currently for normalization output)
   * @param nRows
   *          number of rows in input matrix
   * @param nCols
   *          number of columns in input matrix
   * @param nPCs
   *          number of desired principal components
   * @param splitFactor
   *          divide the block size by this number to increase parallelism
   * @param round
   *          the initial round index, used for naming each round output
   * @param LAST_ROUND
   *          the index of the last round
   * @param sampleRate
   *          if < 1, the input is sampled during normalization
   * @return the error
   * @throws Exception
   */
  double runMapReduce(Configuration conf, DistributedRowMatrix distY,
      InitialValues initVal, Path output, final int nRows, final int nCols,
      final int nPCs, final int splitFactor, final float errSampleRate, final int LAST_ROUND,
      final int normalize) throws Exception {
    int round = 0;
    //The two PPCA variables that improve over each iteration
    double ss = initVal.ss;
    Matrix centralC = initVal.C;
    //initial CtC
    Matrix centralCtC = centralC.transpose().times(centralC);
    final float threshold = 0.00001f;
    int sampleRate=1;
    //1. compute mean and span
    DenseVector ym = new DenseVector(distY.numCols()); //ym=mean(distY)
    MeanAndSpanJob masJob = new MeanAndSpanJob();
    boolean normalizeMean=false;
    if (normalize==1)
    	normalizeMean=true;
    Path meanSpanPath = masJob.compuateMeanAndSpan(distY.getRowPath(),
        output, ym, normalizeMean, conf, ""+round+"-init");
    Path normalizedYPath=null;
    
    //2. normalize the input matrix Y
    if(normalize==1)
    {
	   
	    NormalizeJob normalizeJob = new NormalizeJob();
	    normalizedYPath = normalizeJob.normalize(conf, distY.getRowPath(),
	        meanSpanPath, output, sampleRate, ""+round+"-init");
	    distY = new DistributedRowMatrix(normalizedYPath, getTempPath(), nRows,
	        nCols);
	    distY.setConf(conf);
	    //After normalization, set the split factor
	    if (splitFactor > 1) {
	      FileSystem fss = FileSystem.get(normalizedYPath.toUri(), conf);
	      long blockSize = fss.getDefaultBlockSize() / splitFactor;
	      conf.set("mapred.max.split.size", Long.toString(blockSize));
	    }
	}
    if(normalizedYPath==null)
    	normalizedYPath=distY.getRowPath();
    
    //3. compute the 2-norm of Y
    Norm2Job normJob = new Norm2Job();
    double norm2 = normJob.computeFNorm(conf, normalizedYPath,
        meanSpanPath, getTempPath(), ""+round+"-init");
    if (sampleRate < 1) { // rescale
      norm2 = norm2 / sampleRate;
    }
    
    DenseVector xm = new DenseVector(nPCs);
    log.info("SSSSSSSSSSSSSSSSSSSSSSSSSSSS " + ss);
    DistributedRowMatrix distY2X = null;
    DistributedRowMatrix distC = null;
    double prevObjective = Double.MAX_VALUE;
    double error = 0;
    double relChangeInObjective = Double.MAX_VALUE;
    for (; (round < LAST_ROUND && relChangeInObjective > threshold); round ++) {
      // Sx = inv( ss * eye(d) + CtC );
      Matrix centralSx = centralCtC.clone();
      centralSx.viewDiagonal().assign(Functions.plus(ss));
      centralSx = inv(centralSx);
      // X = Y * C * Sx' => Y2X = C * Sx'
      Matrix centralY2X = centralC.times(centralSx.transpose());
      distY2X = PCACommon.toDistributedRowMatrix(centralY2X, getTempPath(),
          getTempPath(), "CSxt" + round);
      // Xm = Ym * Y2X
      PCACommon.denseVectorTimesMatrix(ym, centralY2X, xm);
      // We skip computing X as we generate it on demand using Y and Y2X

      //Compute X'X and Y'X 
      CompositeJob compositeJob = new CompositeJob();
      compositeJob.computeYtXandXtX(distY, distY2X, ym, xm, getTempPath(), conf,
          ""+round);
      Matrix centralXtX = compositeJob.xtx;
      Matrix centralYtX = compositeJob.ytx;
      if (sampleRate < 1) { // rescale
        centralXtX.assign(Functions.div(sampleRate));
        centralYtX.assign(Functions.div(sampleRate));
      }

      // XtX = X'*X + ss * Sx
      final double finalss = ss;
      centralXtX.assign(centralSx, new DoubleDoubleFunction() {
        @Override
        public double apply(double arg1, double arg2) {
          return arg1 + finalss * arg2;
        }
      });
      // C = (Ye'*X) / SumXtX;
      Matrix invXtX_central = inv(centralXtX);
      centralC = centralYtX.times(invXtX_central);
      distC = PCACommon.toDistributedRowMatrix(centralC, getTempPath(),
          getTempPath(), "C" + round);
      centralCtC = centralC.transpose().times(centralC);
      
      // Compute new value for ss
      // ss = ( sum(sum(Ye.^2)) + PCACommon.trace(XtX*CtC) - 2sum(XiCtYit) )
      // /(N*D);
      double ss2 = PCACommon.trace(centralXtX.times(centralCtC));
      VarianceJob varianceJob = new VarianceJob();
      double xctyt = varianceJob.computeVariance(distY, ym, distY2X, xm, distC,
          getTempPath(), conf, "" + round);
      if (sampleRate < 1) { // rescale
        xctyt = xctyt / sampleRate;
      }
      ss = (norm2 + ss2 - 2 * xctyt) / (nRows * nCols);
      log.info("SSSSSSSSSSSSSSSSSSSSSSSSSSSS " + ss + " (" + norm2 + " + "
          + ss2 + " -2* " + xctyt);
      double traceSx = PCACommon.trace(centralSx);
      double traceXtX = PCACommon.trace(centralXtX);
      double traceC = PCACommon.trace(centralC);
      double traceCtC = PCACommon.trace(centralCtC);
      log.info("TTTTTTTTTTTTTTTTT " + traceSx + " " + traceXtX + " "
          + traceC + " " + traceCtC);

      double objective = ss;
      relChangeInObjective = Math.abs(1 - objective / prevObjective);
      prevObjective = objective;
      log.info("Objective:  %.6f    relative change: %.6f \n", objective,
          relChangeInObjective);
      if (!CALCULATE_ERR_ATTHEEND) {
        log.info("Computing the error at round " + round + " ...");
        ReconstructionErrJob errJob = new ReconstructionErrJob();
        error = errJob.reconstructionErr(distY, distY2X, distC, centralC, ym,
            xm, errSampleRate, conf, getTempPath(), "" + round);
        log.info("... end of computing the error at round " + round);
      }
    }

    if (CALCULATE_ERR_ATTHEEND) {
      log.info("Computing the error at round " + round + " ...");
      ReconstructionErrJob errJob = new ReconstructionErrJob();
      error = errJob.reconstructionErr(distY, distY2X, distC, centralC, ym,
          xm, errSampleRate, conf, getTempPath(), "" + round);
      log.info("... end of computing the error at round " + round);
    }

    initVal.C = centralC;
    initVal.ss = ss;
    return error;
  }

  static class InitialValues {
    Matrix C;
    double ss;

    InitialValues(Matrix C, double ss) {
      this.C = C;
      this.ss = ss;
    }
  }

  /***
   * PPCA: sequential PPCA based on the paper from Tipping and Bishop
   * 
   * @param conf
   *          the configuration
   * @param input
   *          the path to the input matrix Y
   * @param output
   *          the output path (not used currently)
   * @param nRows
   *          number or rows in Y
   * @param nCols
   *          number of columns in Y
   * @param nPCs
   *          number of desired principal components
   * @return the error
   * @throws Exception
   */
  double runSequential(Configuration conf, Path input, Path output,
      final int nRows, final int nCols, final int nPCs) throws Exception {
    Matrix centralY = new DenseMatrix(nRows, nCols);
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    if (fs.listStatus(input).length == 0) {
      System.err.println("No file under " + input);
      return 0;
    }
    int row = 0;
    for (VectorWritable vw : new SequenceFileDirValueIterable<VectorWritable>(
        input, PathType.LIST, null, conf)) {
      centralY.assignRow(row, vw.get());
      row++;
    }
    Matrix centralC = PCACommon.randomMatrix(nCols, nPCs);
    double ss = PCACommon.randSS();
    InitialValues initVal = new InitialValues(centralC, ss);
    // Matrix sampledYe = sample(centralY);
    // runSequential(conf, sampledYe, initVal, 100);
    double error = runSequential(conf, centralY, initVal, 100);
    return error;
  }

  /**
   * Run PPCA sequentially given the small input Y which fit into memory This
   * could be used also on sampled data from a distributed matrix
   * 
   * Note: this implementation ignore NaN values by replacing them with 0
   * 
   * @param conf
   *          the configuration
   * @param centralY
   *          the input matrix
   * @param initVal
   *          the initial values for C and ss
   * @param MAX_ROUNDS
   *          maximum number of iterations
   * @return the error
   * @throws Exception
   */
  double runSequential(Configuration conf, Matrix centralY, InitialValues initVal,
      final int MAX_ROUNDS) throws Exception {
    Matrix centralC = initVal.C;
    double ss = initVal.ss;
    final int nRows = centralY.numRows();
    final int nCols = centralY.numCols();
    final int nPCs = centralC.numCols();
    final float threshold = 0.00001f;

    log.info("tracec= " + PCACommon.trace(centralC));
    //ignore NaN elements by replacing them with 0
    for (int r = 0; r < nRows; r++)
      for (int c = 0; c < nCols; c++)
        if (new Double(centralY.getQuick(r, c)).isNaN()) {
          centralY.setQuick(r, c, 0);
        }

    //centralize and normalize the input matrix
    Vector mean = centralY.aggregateColumns(new VectorFunction() {
      @Override
      public double apply(Vector v) {
        return v.zSum() / nRows;
      }
    });
    //also normalize the matrix by dividing each element by its columns range
    Vector spanVector = new DenseVector(nCols);
    for (int c = 0; c < nCols; c++) {
      Vector col = centralY.viewColumn(c);
      double max = col.maxValue();
      double min = col.minValue();
      double span = max - min;
      spanVector.setQuick(c, span);
    }
    for (int r = 0; r < nRows; r++)
      for (int c = 0; c < nCols; c++)
        centralY.set(r, c, (centralY.get(r, c) - mean.get(c))
            / (spanVector.getQuick(c) != 0 ? spanVector.getQuick(c) : 1));
    
    Matrix centralCtC = centralC.transpose().times(centralC);
    log.info("tracectc= " + PCACommon.trace(centralCtC));
    log.info("traceinvctc= " + PCACommon.trace(inv(centralCtC)));
    log.info("traceye= " + PCACommon.trace(centralY));
    log.info("SSSSSSSSSSSSSSSSSSSSSSSSSSSS " + ss);

    int count = 1;
    // old = Inf;
    double old = Double.MAX_VALUE;
    // -------------------------- EM Iterations
    // while count
    Matrix centralX = null;
    int round = 0;
    while (round < MAX_ROUNDS && count > 0) {
      round++;
      // Sx = inv( eye(d) + CtC/ss );
      Matrix Sx = eye(nPCs).times(ss).plus(centralCtC);
      Sx = inv(Sx);
      // X = Ye*C*(Sx/ss);
      centralX = centralY.times(centralC).times(Sx.transpose());
      // XtX = X'*X + ss * Sx;
      Matrix centralXtX = centralX.transpose().times(centralX).plus(Sx.times(ss));
      // C = (Ye'*X) / XtX;
      Matrix tmpInv = inv(centralXtX);
      centralC = centralY.transpose().times(centralX).times(tmpInv);
      // CtC = C'*C;
      centralCtC = centralC.transpose().times(centralC);
      // ss = ( sum(sum( (X*C'-Ye).^2 )) + trace(XtX*CtC) - 2*xcty ) /(N*D);
      double norm2 = centralY.clone().assign(new DoubleFunction() {
        @Override
        public double apply(double arg1) {
          return arg1 * arg1;
        }
      }).zSum();
      ss = norm2 + PCACommon.trace(centralXtX.times(centralCtC));
      //ss3 = sum (X(i:0) * C' * Y(i,:)')
      DenseVector resVector = new DenseVector(nCols);
      double xctyt = 0;
      for (int i = 0; i < nRows; i++) {
        PCACommon.vectorTimesMatrixTranspose(centralX.viewRow(i), centralC,
            resVector);
        double res = resVector.dot(centralY.viewRow(i));
        xctyt += res;
      }
      ss -= 2 * xctyt;
      ss /= (nRows * nCols);

      log.info("SSSSSSSSSSSSSSSSSSSSSSSSSSSS " + ss);
      double traceSx = PCACommon.trace(Sx);
      double traceX = PCACommon.trace(centralX);
      double traceSumXtX = PCACommon.trace(centralXtX);
      double traceC = PCACommon.trace(centralC);
      double traceCtC = PCACommon.trace(centralCtC);
      log.info("TTTTTTTTTTTTTTTTT " + traceSx + " " + traceX + " "
          + traceSumXtX + " " + traceC + " " + traceCtC + " " + 0);

      double objective = ss;
      double rel_ch = Math.abs(1 - objective / old);
      old = objective;
      count++;
      if (rel_ch < threshold && count > 5)
        count = 0;
      log.info("Objective:  %.6f    relative change: %.6f \n", objective,
          rel_ch);
    }

    double norm1Y = centralY.aggregateColumns(new VectorNorm1()).maxValue();
    log.info("Norm1 of Ye is: " + norm1Y);
    Matrix newYerror = centralY.minus(centralX.times(centralC.transpose()));
    double norm1Err = newYerror.aggregateColumns(new VectorNorm1()).maxValue();
    log.info("Norm1 of the reconstruction error is: " + norm1Err);

    initVal.C = centralC;
    initVal.ss = ss;
    return norm1Err / norm1Y;
  }

  /**
   * PPCA: sequential PPCA based on the matlab implementation of Jacob Verbeek
   * 
   * @param conf
   *          the configuration
   * @param input
   *          the path to the input matrix Y
   * @param output
   *          the output path (not used currently)
   * @param nRows
   *          number or rows in Y
   * @param nCols
   *          number of columns in Y
   * @param nPCs
   *          number of desired principal components
   * @return the error
   * @throws Exception
   */
  double runSequential_JacobVersion(Configuration conf, Path input,
      Path output, final int nRows, final int nCols, final int nPCs) throws Exception {
    Matrix centralY = new DenseMatrix(nRows, nCols);
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    if (fs.listStatus(input).length == 0) {
      System.err.println("No file under " + input);
      return 0;
    }
    int row = 0;
    for (VectorWritable vw : new SequenceFileDirValueIterable<VectorWritable>(
        input, PathType.LIST, null, conf)) {
      centralY.assignRow(row, vw.get());
      row++;
    }
    Matrix C = PCACommon.randomMatrix(nCols, nPCs);
    double ss = PCACommon.randSS();
    InitialValues initVal = new InitialValues(C, ss);
    double error = runSequential_JacobVersion(conf, centralY, initVal, 100);
    return error;
  }

  /**
   * Run PPCA sequentially given the small input Y which fit into memory This
   * could be used also on sampled data from a distributed matrix
   * 
   * Note: this implementation ignore NaN values by replacing them with 0
   * 
   * @param conf
   *          the configuration
   * @param centralY
   *          the input matrix
   * @param initVal
   *          the initial values for C and ss
   * @param MAX_ROUNDS
   *          maximum number of iterations
   * @return the error
   * @throws Exception
   */
  double runSequential_JacobVersion(Configuration conf, Matrix centralY,
      InitialValues initVal, final int MAX_ROUNDS) {
    Matrix centralC = initVal.C;// the current implementation doesn't use initial ss of
                         // initVal
    final int nRows = centralY.numRows();
    final int nCols = centralY.numCols();
    final int nPCs = centralC.numCols();
    final float threshold = 0.00001f;

    log.info("tracec= " + PCACommon.trace(centralC));
    // Y = Y - mean(Ye)
    // Also normalize the matrix
    for (int r = 0; r < nRows; r++)
      for (int c = 0; c < nCols; c++)
        if (new Double(centralY.getQuick(r, c)).isNaN()) {
          centralY.setQuick(r, c, 0);
        }
    Vector mean = centralY.aggregateColumns(new VectorFunction() {
      @Override
      public double apply(Vector v) {
        return v.zSum() / nRows;
      }
    });
    Vector spanVector = new DenseVector(nCols);
    for (int c = 0; c < nCols; c++) {
      Vector col = centralY.viewColumn(c);
      double max = col.maxValue();
      double min = col.minValue();
      double span = max - min;
      spanVector.setQuick(c, span);
    }
    for (int r = 0; r < nRows; r++)
      for (int c = 0; c < nCols; c++)
        centralY.set(r, c, (centralY.get(r, c) - mean.get(c))
            / (spanVector.getQuick(c) != 0 ? spanVector.getQuick(c) : 1));

    // -------------------------- initialization
    // CtC = C'*C;
    Matrix centralCtC = centralC.transpose().times(centralC);
    log.info("tracectc= " + PCACommon.trace(centralCtC));
    log.info("traceinvctc= " + PCACommon.trace(inv(centralCtC)));
    log.info("traceye= " + PCACommon.trace(centralY));
    // X = Ye * C * inv(CtC);
    Matrix centralX = centralY.times(centralC).times(inv(centralCtC));
    log.info("tracex= " + PCACommon.trace(centralX));
    // recon = X * C';
    Matrix recon = centralX.times(centralC.transpose());
    log.info("tracerec= " + PCACommon.trace(recon));
    // ss = sum(sum((recon-Ye).^2)) / (N*D-missing);
    double ss = recon.minus(centralY).assign(new DoubleFunction() {
      @Override
      public double apply(double arg1) {
        return arg1 * arg1;
      }
    }).zSum() / (nRows * nCols);
    log.info("SSSSSSSSSSSSSSSSSSSSSSSSSSSS " + ss);

    int count = 1;
    // old = Inf;
    double old = Double.MAX_VALUE;
    // -------------------------- EM Iterations
    // while count
    int round = 0;
    while (round < MAX_ROUNDS && count > 0) {
      round++;
      // ------------------ E-step, (co)variances
      // Sx = inv( eye(d) + CtC/ss );
      Matrix centralSx = eye(nPCs).plus(centralCtC.divide(ss));
      centralSx = inv(centralSx);
      // ------------------ E-step expected value
      // X = Ye*C*(Sx/ss);
      centralX = centralY.times(centralC).times(centralSx.divide(ss));
      // ------------------ M-step
      // SumXtX = X'*X;
      Matrix centralSumXtX = centralX.transpose().times(centralX);
      // C = (Ye'*X) / (SumXtX + N*Sx );
      Matrix tmpInv = inv(centralSumXtX.plus(centralSx.times(nRows)));
      centralC = centralY.transpose().times(centralX).times(tmpInv);
      // CtC = C'*C;
      centralCtC = centralC.transpose().times(centralC);
      // ss = ( sum(sum( (X*C'-Ye).^2 )) + N*sum(sum(CtC.*Sx)) +
      // missing*ss_old ) /(N*D);
      recon = centralX.times(centralC.transpose());
      double error = recon.minus(centralY).assign(new DoubleFunction() {
        @Override
        public double apply(double arg1) {
          return arg1 * arg1;
        }
      }).zSum();
      ss = error + nRows * dot(centralCtC.clone(), centralSx).zSum();
      ss /= (nRows * nCols);

      log.info("SSSSSSSSSSSSSSSSSSSSSSSSSSSS " + ss);
      double traceSx = PCACommon.trace(centralSx);
      double traceX = PCACommon.trace(centralX);
      double traceSumXtX = PCACommon.trace(centralSumXtX);
      double traceC = PCACommon.trace(centralC);
      double traceCtC = PCACommon.trace(centralCtC);
      log.info("TTTTTTTTTTTTTTTTT " + traceSx + " " + traceX + " "
          + traceSumXtX + " " + traceC + " " + traceCtC + " " + 0);

      // objective = N*D + N*(D*log(ss) +PCACommon.trace(Sx)-log(det(Sx)) )
      // +PCACommon.trace(SumXtX) -missing*log(ss_old);
      double objective = nRows * nCols + nRows
          * (nCols * Math.log(ss) + PCACommon.trace(centralSx) - Math
              .log(centralSx.determinant())) + PCACommon.trace(centralSumXtX);
      double rel_ch = Math.abs(1 - objective / old);
      old = objective;
      count++;
      if (rel_ch < threshold && count > 5)
        count = 0;
      System.out.printf("Objective:  %.6f    relative change: %.6f \n",
          objective, rel_ch);
    }

    double norm1Y = centralY.aggregateColumns(new VectorNorm1()).maxValue();
    log.info("Norm1 of Y is: " + norm1Y);
    Matrix newYerror = centralY.minus(centralX.times(centralC.transpose()));
    double norm1Err = newYerror.aggregateColumns(new VectorNorm1()).maxValue();
    log.info("Norm1 of the reconstruction error is: " + norm1Err);

    initVal.C = centralC;
    initVal.ss = ss;
    return norm1Err / norm1Y;
  }

  private static class VectorNorm1 implements VectorFunction {
    @Override
    public double apply(Vector f) {
      return f.norm(1);// sum
    }
  }

  private static Matrix dot(Matrix a, Matrix b) {
    return a.assign(b, new DoubleDoubleFunction() {
      @Override
      public double apply(double arg1, double arg2) {
        return arg1 * arg2;
      }
    });
  }

  private static Matrix inv(Matrix m) {
    // assume m is square
    QRDecomposition qr = new QRDecomposition(m);
    Matrix i = eye(m.numRows());
    Matrix res = qr.solve(i);
    Matrix densRes = toDenseMatrix(res); // to go around sparse matrix bug
    return densRes;
  }

  private static DenseMatrix toDenseMatrix(Matrix origMtx) {
    DenseMatrix mtx = new DenseMatrix(origMtx.numRows(), origMtx.numCols());
    Iterator<MatrixSlice> sliceIterator = origMtx.iterateAll();
    while (sliceIterator.hasNext()) {
      MatrixSlice slice = sliceIterator.next();
      mtx.viewRow(slice.index()).assign(slice.vector());
    }
    return mtx;
  }

  private static Matrix eye(int n) {
    Matrix m = new DenseMatrix(n, n);
    m.assign(0);
    m.viewDiagonal().assign(1);
    return m;
  }

  // utility methods for sampling a matrix
  /**
   * The rate that is used for sampling data
   */
  static float SAMPLE_RATE = 0.10f; // 10% default

  private static void setSampleRate(int numRows, int numCols) {
    final int MEMSPACE = (int) Math.pow(2, 24);// 16M cells
    int sampleRows = MEMSPACE / numCols; // TODO: MEMSPACE < numCols
    SAMPLE_RATE = sampleRows / (float) numRows;
    if (SAMPLE_RATE > 1)
      SAMPLE_RATE = 1;
    log.info("SSSSSSSSSSSample rate: " + SAMPLE_RATE);
  }

  static Matrix sample(DistributedRowMatrix bigMatrix) {
    setSampleRate(bigMatrix.numRows(), bigMatrix.numCols());
    Matrix sampleMatrix = new DenseMatrix(
        (int) (bigMatrix.numRows() * SAMPLE_RATE), bigMatrix.numCols());
    sample(bigMatrix, sampleMatrix);
    return sampleMatrix;
  }

  static Matrix sample(Matrix bigMatrix) {
    setSampleRate(bigMatrix.numRows(), bigMatrix.numCols());
    Matrix sampleMatrix = bigMatrix.like(
        (int) (bigMatrix.numRows() * SAMPLE_RATE), bigMatrix.numCols());
    sample(bigMatrix, sampleMatrix);
    return sampleMatrix;
  }

  static <M extends VectorIterable> Matrix sample(M bigMatrix,
      Matrix sampleMatrix) {
    log.info("Sampling a " + bigMatrix.numRows() + "x" + bigMatrix.numCols()
        + " into a " + sampleMatrix.numRows() + "x" + sampleMatrix.numCols());
    int row = 0;
    Iterator<MatrixSlice> sliceIterator = bigMatrix.iterateAll();
    while (sliceIterator.hasNext() && row < sampleMatrix.numRows()) {
      MatrixSlice slice = sliceIterator.next();
      if (!PCACommon.pass(SAMPLE_RATE)) {
        sampleMatrix.viewRow(row).assign(slice.vector());
        row++;
      }
    }
    return sampleMatrix;
  }

  /**
   * @param args
   */
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new SPCADriver(), args);
  }
}
