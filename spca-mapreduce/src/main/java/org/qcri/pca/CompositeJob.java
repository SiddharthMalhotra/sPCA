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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.ComparisonChain;

/**
 * Xc = Yc * MEM
 * 
 * XtX = Xc' * Xc
 * 
 * YtX = Yc' * Xc
 * 
 * It also considers that Y is sparse and receives the mean vectors Ym and Xm
 * separately.
 * 
 * Yc = Y - Ym
 * 
 * Xc = X - Xm
 * 
 * Xc = (Y - Ym) * MEM = Y * MEM - Ym * MEM = X - Xm
 * 
 * XtX = (X - Xm)' * (X - Xm)
 * 
 * YtX = (Y - Ym)' * (X - Xm)
 * 
 * @author maysam yabandeh
 */
public class CompositeJob extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(CompositeJob.class);
  public static final String MATRIXINMEMORY = "matrixInMemory";
  public static final String MATRIXINMEMORYROWS = "memRows";
  public static final String MATRIXINMEMORYCOLS = "memCols";
  /**
   * The option specifies the output path to X'X
   */
  public static final String XTXPATH = "xtxPath";
  /**
   * The option specifies the path to Ym Vector
   */
  public static final String YMPATH = "ymPath";
  /**
   * The option specifies the path to Xm Vector
   */
  public static final String XMPATH = "xmPath";

  /**
   * The resulting XtX matrix
   */
  DenseMatrix xtx = null;

  /**
   * The resulting YtX matrix
   */
  DenseMatrix ytx = null;

  public void loadXtX(Path ymPath, int inMemMatrixNumCols,
      Configuration conf) {
    if (xtx != null)
      return;
    Path xtxOutputPath = getXtXPathBasedOnYm(ymPath);
    DistributedRowMatrix xtxDistMtx = new DistributedRowMatrix(xtxOutputPath,
        xtxOutputPath.getParent(), inMemMatrixNumCols, inMemMatrixNumCols);
    xtxDistMtx.setConf(conf);
    xtx = PCACommon.toDenseMatrix(xtxDistMtx);
  }

  public void loadYtX(Path outPath, Path tmpPath, int numRows, int numCols,
      Configuration conf) {
    if (ytx != null)
      return;
    DistributedRowMatrix out = new DistributedRowMatrix(outPath,
        tmpPath, numRows,
        numCols);
    out.setConf(conf);
    ytx = PCACommon.toDenseMatrix(out);
  }

  @Override
  public int run(String[] strings) throws Exception {
    throw new Exception("Unimplemented");
  }

  /**
   * Set the output path for XtX relative to the path of Ym
   * 
   * @param ymPath
   *          the path to Ym
   * @return
   */
  static public Path getXtXPathBasedOnYm(Path ymPath) {
    Path xtxOutputPath = new Path(ymPath.getParent(), ymPath.getName() + "-XtX");
    return xtxOutputPath;
  }

  /**
   * Refer to {@link CompositeJob} for a job description. In short, it does
   * 
   * X = Y * MEM
   * 
   * XtX = (X - Xm)' * (X - Xm)
   *  
   * YtX = (Y - Ym)' * (X - Xm)
   * 
   * @param distMatrixY the input matrix Y
   * @param inMemMatrix the in memory matrix MEM
   * @param ym the mean vector of Y
   * @param xm = ym * MEM
   * @param id the unique id for HDFS output directory
   * @return the XtX and YtX wrapped in a CompositeResult object
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public void computeYtXandXtX(
      DistributedRowMatrix distMatrixY, DistributedRowMatrix inMemMatrix,
      Vector ym, Vector xm, Path tmpPath, Configuration conf, String id) throws IOException,
      InterruptedException, ClassNotFoundException {
    if (distMatrixY.numCols() != inMemMatrix.numRows()) {
      throw new CardinalityException(distMatrixY.numCols(), inMemMatrix.numRows());
    }
    Path outPath = new Path(tmpPath, "Composite"+id);
    Path ymPath = PCACommon.toDistributedVector(ym,
        tmpPath, "ym-compositeJob" + id, conf);
    Path xmPath = PCACommon.toDistributedVector(xm,
        tmpPath, "xm-compositeJob" + id, conf);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    if (!fs.exists(outPath)) {
      run(conf, distMatrixY.getRowPath(), inMemMatrix.getRowPath()
          .toString(), inMemMatrix.numRows(), inMemMatrix.numCols(), ymPath
          .toString(), xmPath.toString(), outPath);
    } else {
      log.warn("----------- Skip Compositejob - already exists: " + outPath);
    }
    
    loadXtX(ymPath, inMemMatrix.numCols(), conf);
    loadYtX(outPath, tmpPath, inMemMatrix.numRows(), inMemMatrix.numCols(), conf);
  }
  
  /**
   * Computes XtX and YtX
   * 
   * Xc = (Y - Ym) * MEM = Y * MEM - Ym * MEM = X - Xm
   * 
   * XtX = (X - Xm)' * (X - Xm) YtX = (Y - Ym)' * (Y - Ym)
   * 
   * @param conf
   *          the configuration
   * @param matrixInputPath
   *          Y
   * @param inMemMatrixDir
   *          MEM, where X = Y * MEM
   * @param inMemMatrixNumRows
   *          MEM.rows
   * @param inMemMatrixNumCols
   *          MEM.cols
   * @param ymPath
   *          Ym
   * @param xmPath
   *          Xm
   * @param matrixOutputPath
   *          YtX
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public void run(Configuration conf, Path matrixInputPath,
      String inMemMatrixDir, int inMemMatrixNumRows, int inMemMatrixNumCols,
      String ymPath, String xmPath, Path matrixOutputPath) throws IOException,
      InterruptedException, ClassNotFoundException {
    conf.set(MATRIXINMEMORY, inMemMatrixDir);
    conf.setInt(MATRIXINMEMORYROWS, inMemMatrixNumRows);
    conf.setInt(MATRIXINMEMORYCOLS, inMemMatrixNumCols);
    conf.set(YMPATH, ymPath);
    conf.set(XMPATH, xmPath);
    Path xtxOutputPath = getXtXPathBasedOnYm(new Path(ymPath));
    conf.set(XTXPATH, xtxOutputPath.toString());
    Job job = new Job(conf);
    job.setJobName("CompositeJob-" + matrixInputPath.getName());
    job.setJarByClass(CompositeJob.class);
    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);
    FileInputFormat.addInputPath(job, matrixInputPath);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setMapperClass(MyMapper.class);
    job.setReducerClass(MyReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapOutputKeyClass(CompositeWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);
    job.setSortComparatorClass(CompositeWritable.class);
    job.setGroupingComparatorClass(CompositeWritable.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.submit();
    job.waitForCompletion(true);
  }

  public static class MyMapper extends
      Mapper<IntWritable, VectorWritable, CompositeWritable, VectorWritable> {
    // input arguments
    private DenseMatrix inMemMatrix;
    private Vector ym;
    private Vector xm;
    // developing variables
    private DenseVector xi;
    private DenseVector sumxi;
    private int totalRows = 0;
    private DenseMatrix ytxMatrix;
    private DenseMatrix xtxMatrix;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      Path inMemMatrixPath = new Path(conf.get(MATRIXINMEMORY));
      int inMemMatrixNumRows = conf.getInt(MATRIXINMEMORYROWS, 0);
      int inMemMatrixNumCols = conf.getInt(MATRIXINMEMORYCOLS, 0);
      Path ymPath = new Path(conf.get(YMPATH));
      Path xmPath = new Path(conf.get(XMPATH));
      try {
        ym = PCACommon.toDenseVector(ymPath, conf);
        xm = PCACommon.toDenseVector(xmPath, conf);
      } catch (IOException e) {
        e.printStackTrace();
      }
      // TODO: add an argument for temp path
      Path tmpPath = inMemMatrixPath.getParent();
      DistributedRowMatrix distMatrix = new DistributedRowMatrix(
          inMemMatrixPath, tmpPath, inMemMatrixNumRows, inMemMatrixNumCols);
      distMatrix.setConf(conf);
      inMemMatrix = PCACommon.toDenseMatrix(distMatrix);
    }

    /**
     * Perform in-memory matrix multiplication xi = yi' * MEM
     */
    @Override
    public void map(IntWritable r, VectorWritable v, Context context)
        throws IOException, InterruptedException {
      Vector yi = v.get();
      if (ytxMatrix == null) {
        ytxMatrix = new DenseMatrix(ym.size(), xm.size());
        xtxMatrix = new DenseMatrix(xm.size(), xm.size() + 1);
        // the last col is row id
        sumxi = new DenseVector(xm.size());
      }

      // 1. Xi = Yi * MEM
      if (xi == null)
        xi = new DenseVector(inMemMatrix.numCols());
      PCACommon.sparseVectorTimesMatrix(yi, inMemMatrix, xi);

      // Sum(Xi)
      sumxi.assign(xi, Functions.PLUS);
      totalRows++;

      // 2. Y' * X ----mapper part
      AtBMapper(yi, ym, xi, xm, ytxMatrix);

      // 3. X' * X ----mapper part
      AtBMapper(xi, xm, xi, xm, xtxMatrix);
    }

    @Override
    public void cleanup(Context context) throws InterruptedException,
        IOException {
      // 2. Y' * X ----combiner part
      AtxBCombiner(ytxMatrix, ym, xm, sumxi, totalRows);
      // 3. X' * X ----combiner part
      AtxBCombiner(xtxMatrix, xm, xm, sumxi, totalRows);

      VectorWritable outVector = new VectorWritable();
      CompositeWritable ytxCompositeKey = new CompositeWritable(
          CompositeWritable.YTX_TYPE);
      for (int i = 0; i < ytxMatrix.numRows(); i++) {
        ytxCompositeKey.set(i);
        outVector.set(ytxMatrix.viewRow(i));
        context.write(ytxCompositeKey, outVector);
      }
      // for all XtX rows the key is the same
      // the last column of the value vector determines the row id
      CompositeWritable xtxSingleKey = new CompositeWritable(
          CompositeWritable.XTX_TYPE);
      int idCol = xtxMatrix.numCols() - 1;// last is id column
      for (int i = 0; i < xtxMatrix.numRows(); i++) {
        xtxMatrix.setQuick(i, idCol, i);
        outVector.set(xtxMatrix.viewRow(i));
        context.write(xtxSingleKey, outVector);
      }
    }

    /***
     * Mi = (Yi-Ym)' x (Xi-Xm) = Yi' x (Xi-Xm) - Ym' x (Xi-Xm)
     * 
     * M = Sum(Mi) = Sum(Yi' x (Xi-Xm)) - Ym' x (Sum(Xi)-N*Xm)
     * 
     * The first part is done in mapper and the second in the combiner
     */
    private void AtBMapper(Vector yi, Vector ym, Vector xi, Vector xm,
        DenseMatrix resMatrix) {
      // 1. Sum(Yi' x (Xi-Xm))
      int xSize = xi.size();
      Iterator<Vector.Element> nonZeroElements = yi.nonZeroes().iterator();
      while (nonZeroElements.hasNext()) {
        Vector.Element e = nonZeroElements.next();
        int yRow = e.index();
        double yScale = e.get();
        for (int xCol = 0; xCol < xSize; xCol++) {
          double centeredValue = xi.getQuick(xCol) - xm.getQuick(xCol);
          double currValue = resMatrix.getQuick(yRow, xCol);
          currValue += centeredValue * yScale;
          resMatrix.setQuick(yRow, xCol, currValue);
        }
      }
    }

    /***
     * Mi = (Yi-Ym)' x (Xi-Xm) = Yi' x (Xi-Xm) - Ym' x (Xi-Xm)
     * 
     * M = Sum(Mi) = Sum(Yi' x (Xi-Xm)) - Ym' x (Sum(Xi)-N*Xm)
     * 
     * The first part is done in mapper and the second in the combiner
     */
    private void AtxBCombiner(DenseMatrix resMatrix, Vector ym, Vector xm,
        Vector xsum, int nRows) {
      // 2. - Ym' x (Sum(Xi)-N*Xm)
      int ysize = ym.size();
      int xsize = xsum.size();
      for (int yRow = 0; yRow < ysize; yRow++) {
        double scale = ym.getQuick(yRow);
        for (int xCol = 0; xCol < xsize; xCol++) {
          double centeredValue = xsum.getQuick(xCol) - nRows
              * xm.getQuick(xCol);
          double currValue = resMatrix.getQuick(yRow, xCol);
          currValue -= centeredValue * scale;
          resMatrix.setQuick(yRow, xCol, currValue);
        }
      }
    }
  }

  public static class MyReducer extends
      Reducer<CompositeWritable, VectorWritable, IntWritable, VectorWritable> {
    IntWritable iw = new IntWritable();
    VectorWritable vw = new VectorWritable();
    DenseMatrix xtx = null;
    Path xtxOutputPath;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      xtxOutputPath = new Path(conf.get(XTXPATH));
    }

    @Override
    public void reduce(CompositeWritable compositeId,
        Iterable<VectorWritable> vectors, Context context) throws IOException,
        InterruptedException {
      Iterator<VectorWritable> it = vectors.iterator();
      if (!it.hasNext()) {
        return;
      }
      //All XtX rows are mapped to the same key
      if (compositeId.isXtX()) {
        writeXtXToFile(vectors, xtxOutputPath);
        return;
      }
      //Reduce YtX
      Vector accumulator = it.next().get();
      while (it.hasNext()) {
        Vector row = it.next().get();
        accumulator.assign(row, Functions.PLUS);
      }
      iw.set(compositeId.rowId);
      vw.set(accumulator);
      context.write(iw, vw);
    }

    private void writeXtXToFile(Iterable<VectorWritable> vectors,
        Path xtxOutputPath) throws IOException {
      if (xtx != null)
        throw new IOException(
            "Error: second call to xtx writer at CompositeJob");
      Configuration conf = new Configuration();
      FileSystem fs = FileSystem.get(xtxOutputPath.toUri(), conf);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf,
          xtxOutputPath, IntWritable.class, VectorWritable.class);
      try {
        for (VectorWritable v : vectors) {
          Vector vector = v.get();
          if (xtx == null)
            xtx = new DenseMatrix(vector.size() - 1, vector.size() - 1);
          int idCol = vector.size() - 1;// last is id column
          int id = (int) (vector.get(idCol));
          // exclude the id column
          xtx.viewRow(id).assign(vector.viewPart(0, vector.size() - 1),
              Functions.PLUS);
        }
        for (int i = 0; i < xtx.numRows(); i++) {
          iw.set(i);
          vw.set(xtx.viewRow(i));
          writer.append(iw, vw);
        }
      } finally {
        writer.close();
      }
    }
  }

  /**
   * Composite key that allows sending two types of entries to the reducers one
   * for XtX and one for YtX
   */
  static class CompositeWritable extends WritableComparator implements
      WritableComparable<CompositeWritable> {
    public static final byte XTX_TYPE = 0;
    public static final byte YTX_TYPE = 1;
    private byte type;
    private int rowId;

    public CompositeWritable() {
      super(CompositeWritable.class);
    }

    public CompositeWritable(byte type) {
      this();
      this.type = type;
    }

    public void set(int rowId) {
      this.rowId = rowId;
    }

    public boolean isXtX() {
      return type == XTX_TYPE;
    }

    @Override
    public void readFields(DataInput in) throws IOException {
      type = in.readByte();
      rowId = in.readInt();
    }

    @Override
    public void write(DataOutput out) throws IOException {
      out.writeByte(type);
      out.writeInt(rowId);
    }

    @Override
    public int compareTo(CompositeWritable o) {
      if (type == XTX_TYPE && type == o.type)
        return 0;// make a single reducer take care of XtX
      return ComparisonChain.start().compare(type, o.type)
          .compare(rowId, o.rowId).result();
    }

    @Override
    public int hashCode() {
      if (type == XTX_TYPE)
        return 0;// make a single reducer take care of XtX
      return rowId + 1; // skip 0
    }

    @Override
    public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
      byte t1 = (byte) b1[s1];
      byte t2 = (byte) b2[s2];
      if (t1 == XTX_TYPE && t1 == t2)
        return 0;
      if (t1 != t2)
        return t1 - t2;
      s1++;
      s2++;

      int r1 = readInt(b1, s1);
      int r2 = readInt(b2, s2);
      return r1 - r2;
    }
    
    @Override
    public String toString() {
      String typeStr = type == XTX_TYPE ? "xtx-" : "ytx-";
      return typeStr += rowId;
    }
  }

}
