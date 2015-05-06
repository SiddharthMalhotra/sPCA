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

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * @author maysam yabandeh
 */
public class PrepareInput {

	/**
	 * Generate SequnceFile format from a text file
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		if (args.length < 1) {
			System.err.println("input text file is missing");
			return;
		}
		textToSequnceFile(args[0]);
	}

	private static void textToSequnceFile(String inputStr) throws IOException {
		BufferedReader inputReader = new BufferedReader(
				new FileReader(inputStr));
		Configuration conf = new Configuration();
		Path inputPath = new Path(inputStr);
		Path outputPath = new Path(inputPath.getParent(), inputPath.getName()
				+ ".formatted");
		FileSystem fs = FileSystem.get(inputPath.toUri(), conf);
		SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf,
				outputPath, IntWritable.class, VectorWritable.class);
		VectorWritable vectorWritable = new VectorWritable();
		String line;
		int index = 0;
		try {
			while ((line = inputReader.readLine()) != null) {
				String[] columns = line.split(" ");
				int shift = 0;
				if (columns[0].isEmpty())
					shift++;
				double[] columnsDouble = new double[columns.length-shift];
				for (int i = 0; i < columnsDouble.length; i++) {
					columnsDouble[i] = Double.valueOf(columns[i+shift]);
				}
				Vector vector = new DenseVector(columnsDouble, true);
				vectorWritable.set(vector);
				writer.append(new IntWritable(index), vectorWritable);
				index++;
			}
		} finally {
			writer.close();
		}
		inputReader.close();
		System.out.println("Finish writing to " + outputPath);
	}

}
