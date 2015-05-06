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
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;

/**
 * @author maysam yabandeh
 */
public class TestSequenceFile {

  /**
   * Generate SequnceFile format from a text file
   * 
   * @param args
   * @throws IOException
   */
  public static void main(String[] args) throws IOException {
    if (args.length < 1) {
      System.err.println("input text file is missing");
      return;
    }
    printSequenceFile(args[0], Integer.parseInt(args.length > 1 ? args[1] : "0"));
  }

  private static void printSequenceFile(String inputStr, int printRow) throws IOException {
    Configuration conf = new Configuration();
    Path finalNumberFile = new Path(inputStr);
    SequenceFile.Reader reader = new SequenceFile.Reader(FileSystem.get(conf),
        finalNumberFile, conf);
    IntWritable key = new IntWritable();
    VectorWritable value = new VectorWritable();
    Vector printVector = null;
    while (reader.next(key, value)) {
      if (key.get() == printRow)
        printVector = value.get();
      int cnt = 0;
      Iterator<Element> iter = value.get().nonZeroes().iterator();
      for (; iter.hasNext(); iter.next())
        cnt++;
      System.out.println("# "+ key + " " + cnt + " " + value.get().zSum());
    }
    reader.close();
    if (printVector != null)
      System.out.println("##### "+ printRow + " " + printVector);
    else
      System.out.println("##### "+ key + " " + value.get());
  }
}



























