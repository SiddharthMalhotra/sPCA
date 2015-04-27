package org.qcri.sparkpca;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class FileFormat {
	 private final static Logger log = LoggerFactory.getLogger(FileFormat.class);
	 public enum OutputFormat {
		DENSE,  //Dense matrix 
		LIL, //List of lists
		COO, //Coordinate List
	 } 
	 public enum InputFormat {
		DENSE,
		LIL,
		COO
	 } 
	 public static void main(String[] args) {
		
	}
	public static void convertFromDenseToSeq(String inputPath)
	{
		
	}
	public static void convertFromCooToSeq(String inputPath, String outputPath)
	{
		
	}
}
