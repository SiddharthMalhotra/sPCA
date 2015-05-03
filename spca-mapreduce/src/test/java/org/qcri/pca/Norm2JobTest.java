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
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.qcri.pca.DummyRecordWriter;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;

/**
 * @author maysam yabandeh
 */
public class Norm2JobTest extends PCATestCase {

  // try different combination of 0, NaN, +, -
  private final double[][] inputVectors = {
      { -0.49, Double.NaN, 0, -0.49, 3, -13.3 },
      { Double.NaN, -0.48, 0.45, 0.54, -20, 3.2 },
      { 0, 0.71, 0, 0.08, -12.2, 0.03 }, { 0.13, -0.06, -0.72, 0, 4, -0.00003 } };
  int cols = inputVectors[0].length;
  int rows = inputVectors.length;
  private Configuration conf;
  private Path meanSpanFilePath;

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
    Path meanSpanDirPath = new Path("/tmp/" + currTime + "/meanSpan");
    meanSpanFilePath = new MeanAndSpanJob().getMeanSpanPath(meanSpanDirPath);
    FileSystem fs;
    try {
      fs = FileSystem.get(meanSpanDirPath.toUri(), conf);
      fs.mkdirs(meanSpanDirPath);
      fs.deleteOnExit(meanSpanDirPath);
    } catch (IOException e) {
      e.printStackTrace();
      Assert.fail("Error in creating meanSpan direcoty " + meanSpanDirPath);
      return;
    }
    prepareTheMeanSpanFile(fs);
  }

  /**
   * prepares the meanspan file by running mapper and reducer of MeanAndSpanJob
   * 
   * @throws Exception
   */
  void prepareTheMeanSpanFile(FileSystem fs) throws Exception {
    Array2DenseVector array2Vector = new Array2DenseVector();
    Configuration conf = new Configuration();
    MeanAndSpanJob.MeanAndSpanMapper mapper = new MeanAndSpanJob.MeanAndSpanMapper();
    // construct the writers
    DummyRecordWriter<IntWritable, VectorWritable> mapWriter = new DummyRecordWriter<IntWritable, VectorWritable>();
    Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable>.Context context = DummyRecordWriter
        .build(mapper, conf, mapWriter);
    // perform the mapping
    for (int i = 0; i < inputVectors.length; i++) {
      VectorWritable row = new VectorWritable(array2Vector.get(inputVectors[i]));
      mapper.map(new IntWritable(i), row, context);
    }
    mapper.cleanup(context);

    MeanAndSpanJob.MeanAndSpanReducer reducer = new MeanAndSpanJob.MeanAndSpanReducer();
    // set up the writers
    final SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf,
        meanSpanFilePath, IntWritable.class, VectorWritable.class);
    RecordWriter<IntWritable, VectorWritable> redWriter = new RecordWriter<IntWritable, VectorWritable>() {
      @Override
      public void close(TaskAttemptContext arg0) throws IOException,
          InterruptedException {
      }

      @Override
      public void write(IntWritable arg0, VectorWritable arg1)
          throws IOException, InterruptedException {
        writer.append(arg0, arg1);
      }
    };
    Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable>.Context redContext = DummyRecordWriter
        .build(reducer, conf, redWriter, IntWritable.class,
            VectorWritable.class);
    // perform the reduction
    for (IntWritable key : mapWriter.getKeys()) {
      reducer.reduce(key, mapWriter.getValue(key), redContext);
    }
    reducer.cleanup(redContext);
    writer.close();
  }

  @Test
  public void testWithDenseVectors() throws Exception {
    DummyRecordWriter<IntWritable, VectorWritable> normalizeOutput = normalize(new Array2DenseVector());
    testJob(normalizeOutput);
  }

  @Test
  public void testWithSparseVectors() throws Exception {
    DummyRecordWriter<IntWritable, VectorWritable> normalizeOutput = normalize(new Array2SparseVector());
    testJob(normalizeOutput);
  }

  DummyRecordWriter<IntWritable, VectorWritable> normalize(
      Array2Vector array2Vector) throws Exception {
    Configuration conf = new Configuration();
    conf.set(NormalizeJob.MEANSPANOPTION, meanSpanFilePath.toString());
    NormalizeJob.NormalizeMapper mapper = new NormalizeJob.NormalizeMapper();
    // construct the writers
    DummyRecordWriter<IntWritable, VectorWritable> mapWriter = new DummyRecordWriter<IntWritable, VectorWritable>();
    Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable>.Context context = DummyRecordWriter
        .build(mapper, conf, mapWriter);
    // perform the mapping
    mapper.setup(context);
    for (int i = 0; i < inputVectors.length; i++) {
      VectorWritable row = new VectorWritable(array2Vector.get(inputVectors[i]));
      mapper.map(new IntWritable(i), row, context);
    }
    return mapWriter;
  }

  void testJob(DummyRecordWriter<IntWritable, VectorWritable> normalizeOutput)
      throws Exception {
    Configuration conf = new Configuration();
    conf.set(Norm2Job.MEANSPANOPTION, meanSpanFilePath.toString());
    Norm2Job.MyMapper mapper = new Norm2Job.MyMapper();
    // construct the writers
    DummyRecordWriter<NullWritable, DoubleWritable> mapWriter = new DummyRecordWriter<NullWritable, DoubleWritable>();
    Mapper<IntWritable, VectorWritable, NullWritable, DoubleWritable>.Context context = DummyRecordWriter
        .build(mapper, conf, mapWriter);
    // perform the mapping
    mapper.setup(context);
    for (IntWritable key : normalizeOutput.getKeys()) {
      List<VectorWritable> list = normalizeOutput.getValue(key);
      assertEquals("Normalizer made more than one value for each key!", 1,
          list.size());
      VectorWritable vw = list.get(0);
      mapper.map(key, vw, context);
    }
    mapper.cleanup(context);

    Norm2Job.MyReducer reducer = new Norm2Job.MyReducer();
    // set up the writers
    DummyRecordWriter<NullWritable, DoubleWritable> redWriter = new DummyRecordWriter<NullWritable, DoubleWritable>();
    Reducer<NullWritable, DoubleWritable, NullWritable, DoubleWritable>.Context redContext = DummyRecordWriter
        .build(reducer, conf, redWriter, NullWritable.class,
            DoubleWritable.class);
    // perform the reduction
    for (NullWritable key : mapWriter.getKeys()) {
      reducer.reduce(key, mapWriter.getValue(key), redContext);
    }
    verifyReducerOutput(redWriter);
  }

  private void verifyReducerOutput(
      DummyRecordWriter<NullWritable, DoubleWritable> writer) {
    Assert.assertEquals("The reducer should output one keys!", 1, writer
        .getKeys().size());
    double norm2 = norm2(inputVectors);
    for (NullWritable key : writer.getKeys()) {
      List<DoubleWritable> list = writer.getValue(key);
      assertEquals("Reducer did not combine the results!", 1, list.size());
      Double v = list.get(0).get();
      Assert.assertEquals("The 2norm is incorrect: ", norm2, v, EPSILON);
    }
  }

  private double norm2(double[][] vectors) {
    double[][] normalizedVectors = vectors.clone();
    for (int i = 0; i < vectors.length; i++)
      normalizedVectors[i] = vectors[i].clone();
    vectors = normalizedVectors;
    // 1. normalize it
    for (int c = 0; c < cols; c++) {
      double min = Double.MAX_VALUE;
      double max = Double.MIN_VALUE;
      for (int r = 0; r < rows; r++) {
        double val = NaN2Zero(vectors[r][c]);
        min = Math.min(min, val);
        max = Math.max(max, val);
      }
      double span = max - min;
      for (int r = 0; r < rows; r++) {
        double val = NaN2Zero(vectors[r][c]);
        val = val / (span != 0 ? span : 1);
        vectors[r][c] = val;
      }
    }
    // 2. compute norm2
    double norm2 = 0;
    for (int c = 0; c < cols; c++) {
      double sum = 0;
      for (int r = 0; r < rows; r++) {
        double val = vectors[r][c];
        sum += val;
      }
      double mean = sum / rows;
      for (int r = 0; r < rows; r++) {
        double val = (vectors[r][c] - mean);
        norm2 += (val * val);
      }
    }
    return norm2;
  }

  private double NaN2Zero(double val) {
    return Double.isNaN(val) ? 0 : val;
  }
}
