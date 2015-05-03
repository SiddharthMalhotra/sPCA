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
import java.util.List;

import junit.framework.Assert;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.qcri.pca.DummyRecordWriter;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.junit.Before;
import org.junit.Test;

/**
 * @author maysam yabandeh
 */
public class ReconstructionErrJobTest extends PCATestCase {

  // try different combination of 0, +, -
  // it assumes NaN are removed as part of normalization
  private final double[][] inputVectors = { { -0.49, 0, 3, -13.3 },
      { 0, -0.48, 0.45, 3.2 }, { 0.13, -0.06, 4, -0.00003 } };
  private final double[][] y2xVectors = { { -0.49, 0 }, { 0, -0.48 },
      { 0, 0.71 }, { 4, -0.00003 } };
  final int xsize = y2xVectors[0].length;
  private final double[][] cVectors = { { 0, 0.000302 }, { 0, 453 },
      { -0.0032, -9.9999 }, { -0.00003, 0 } };
  int cols = inputVectors[0].length;
  int rows = inputVectors.length;
  double[] ym, xm, ymC;
  private Configuration conf;
  private Path ymPath;
  private Path zmPath;
  private Path y2xMatrixPath;
  private Path cMatrixPath;
  private double reconstructionError;
  private double yNorm;
  private double centralizedYNorm;

  interface Array2Vector {
    Vector get(double[] array);
  }

  class Array2DenseVector implements Array2Vector {
    public Vector get(double[] array) {
      return new DenseVector(array);
    }
  }

  class Array2SparseVector implements Array2Vector {
    public Vector get(double[] array) {
      return new SequentialAccessSparseVector(new DenseVector(array));
    }
  }

  @Before
  public void setup() throws Exception {
    conf = new Configuration();
    long currTime = System.currentTimeMillis();
    Path outputDir = new Path("/tmp/" + currTime);
    FileSystem fs;
    try {
      fs = FileSystem.get(outputDir.toUri(), conf);
      fs.mkdirs(outputDir);
      fs.deleteOnExit(outputDir);
    } catch (IOException e) {
      e.printStackTrace();
      Assert.fail("Error in creating output direcoty " + outputDir);
      return;
    }
    ym = computeMean(inputVectors);
    double[] xm = new double[xsize];
    times(ym, y2xVectors, xm);
    double[] zm = new double[cols];
    timesTranspose(xm, cVectors, zm);
    for (int c = 0; c < cols; c++)
      zm[c] -= ym[c];
    ymPath = PCACommon.toDistributedVector(new DenseVector(ym), outputDir,
        "ym", conf);
    zmPath = PCACommon.toDistributedVector(new DenseVector(zm), outputDir,
        "zm", conf);
    DistributedRowMatrix distMatrix = PCACommon.toDistributedRowMatrix(
        new DenseMatrix(y2xVectors), outputDir, outputDir, "y2xMatrix");
    y2xMatrixPath = distMatrix.getRowPath();
    distMatrix = PCACommon.toDistributedRowMatrix(
        new DenseMatrix(cVectors), outputDir, outputDir, "cMatrix");
    cMatrixPath = distMatrix.getRowPath();
    computeError(inputVectors);
  }

  @Test
  public void testWithDenseVectors() throws Exception {
    testJob(new Array2DenseVector());
  }

  @Test
  public void testWithSparseVectors() throws Exception {
    testJob(new Array2SparseVector());
  }

  void testJob(Array2Vector array2Vector) throws Exception {
    conf.set(ReconstructionErrJob.MATRIXY2X, y2xMatrixPath.toString());
    conf.set(ReconstructionErrJob.RECONSTRUCTIONMATRIX, cMatrixPath.toString());
    conf.set(ReconstructionErrJob.YMPATH, ymPath.toString());
    conf.set(ReconstructionErrJob.ZMPATH, zmPath.toString());
    conf.setInt(ReconstructionErrJob.YCOLS, cols);
    conf.setInt(ReconstructionErrJob.XCOLS, xsize);
    ReconstructionErrJob.MyMapper mapper = new ReconstructionErrJob.MyMapper();
    // construct the writers
    DummyRecordWriter<IntWritable, VectorWritable> mapWriter = 
        new DummyRecordWriter<IntWritable, VectorWritable>();
    Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable>.Context context = 
        DummyRecordWriter.build(mapper, conf, mapWriter);
    // perform the mapping
    mapper.setup(context);
    for (int i = 0; i < inputVectors.length; i++) {
      VectorWritable row = new VectorWritable(array2Vector.get(inputVectors[i]));
      mapper.map(new IntWritable(i), row, context);
    }
    mapper.cleanup(context);

    // perform the reducing
    ReconstructionErrJob.MyReducer reducer = new ReconstructionErrJob.MyReducer();
    DummyRecordWriter<IntWritable, DoubleWritable> redWriter = 
        new DummyRecordWriter<IntWritable, DoubleWritable>();
    Reducer<IntWritable, VectorWritable, IntWritable, DoubleWritable>.Context redContext =
        DummyRecordWriter.build(reducer, conf, redWriter, IntWritable.class,
            VectorWritable.class);
    for (IntWritable key : mapWriter.getKeys()) {
      reducer.reduce(key, mapWriter.getValue(key), redContext);
    }

    verifyReducerOutput(redWriter);
  }

  private void verifyReducerOutput(
      DummyRecordWriter<IntWritable, DoubleWritable> writer) {
    Assert.assertEquals("The reducer should output three key!", 3, writer
        .getKeys().size());
    for (IntWritable key : writer.getKeys()) {
      List<DoubleWritable> list = writer.getValue(key);
      assertEquals("reducer produces more than one values per key!", 1,
          list.size());
      Double value = list.get(0).get();
      switch (key.get()) {
      case 0:
        assertEquals("the computed reconstructionError is incorrect!",
            reconstructionError, value, EPSILON);
        break;
      case 1:
        assertEquals("the computed yNorm is incorrect!", yNorm, value, EPSILON);
        break;
      case 2:
        assertEquals("the computed centralizedYNorm is incorrect!",
            centralizedYNorm, value, EPSILON);
        break;
      default:
        fail("Unknown key in reading the results: " + key);
      }
    }
  }

  private void computeError(double[][] vectors) {
    double[][] vectorsCopy = vectors.clone();
    for (int i = 0; i < vectors.length; i++)
      vectorsCopy[i] = vectors[i].clone();
    vectors = vectorsCopy;
    yNorm = fnorm(vectors);
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++)
        vectors[r][c] -= ym[c];
    centralizedYNorm = fnorm(vectors);
    double[] xi = new double[xsize];
    double[] newyi = new double[cols];
    for (int r = 0; r < rows; r++) {
      double[] yi = vectors[r];
      times(yi, y2xVectors, xi);
      timesTranspose(xi, cVectors, newyi);
      for (int c = 0; c < cols; c++)
        yi[c] = Math.abs(yi[c] - newyi[c]);
    }
    reconstructionError = fnorm(vectors);
  }

  private double fnorm(double[][] vectors) {
    double max = Double.MIN_VALUE;
    for (int c = 0; c < cols; c++) {
      double sum = 0;
      for (int r = 0; r < rows; r++)
        sum += Math.abs(vectors[r][c]);
      max = Math.max(max, sum);
    }
    return max;
  }

  private double[] computeMean(double[][] vectors) {
    double[] meanVector = new double[cols];
    for (int c = 0; c < cols; c++) {
      double sum = 0;
      for (int r = 0; r < rows; r++) {
        double val = vectors[r][c];
        sum += val;
      }
      meanVector[c] = sum / rows;
    }
    return meanVector;
  }

  private void times(double[] ym, double[][] matrix, double[] resVector) {
    int maxC = matrix[0].length;
    int maxR = matrix.length;
    for (int inMemC = 0; inMemC < maxC; inMemC++) {
      double sum = 0;
      for (int r = 0; r < maxR; r++) {
        double val = ym[r];
        double inMemVal = matrix[r][inMemC];
        sum += val * inMemVal;
      }
      resVector[inMemC] = sum;
    }
  }

  private void timesTranspose(double[] vector, double[][] matrix, double[] resVector) {
    int maxC = matrix[0].length;
    int maxR = matrix.length;
    for (int inMemR = 0; inMemR < maxR; inMemR++) {
      double sum = 0;
      for (int c = 0; c < maxC; c++) {
        double val = vector[c];
        double inMemVal = matrix[inMemR][c];
        sum += val * inMemVal;
      }
      resVector[inMemR] = sum;
    }
  }
  
}
