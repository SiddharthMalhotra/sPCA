/**
 * QCRI, sPCA LICENSE
 * sPCA is a scalable implementation of Principal Component Analysis (PCA) on of Spark and MapReduce
 *
 * Copyright (c) 2015, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
*/

package org.qcri.sparkpca;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.QRDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.qcri.sparkpca.FileFormat.OutputFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.Tuple2;

import com.esotericsoftware.minlog.Log;


/**
 * This class includes the utility functions that is used by PCA algorithm
 * 
 * 
 * @author Tarek Elgamal
 */
class PCAUtils {
	 private final static Logger log = LoggerFactory.getLogger(SparkPCA.class);
	
	/**
	 * We use a single random object to help reproducing the erroneous scenarios
	 */
	static Random random = new Random(0);


	

	/**
	   * @return random initialization for variance
	*/
	static double randSS() {
		return random.nextDouble();
	}
	
	
	/**
	 * @return hard-coded initialization variance used for validation 
	 */
	static double randValidationSS() {
		// return random.nextDouble();
		return 0.9644868606768501;
	}
	
	/**
	 * A randomly initialized matrix
	 * 
	 * @param rows
	 * @param cols
	 * @return
	 */

	static Matrix randomMatrix(int rows, int cols) {
		Matrix randM = new DenseMatrix(rows, cols);
		randM.assign(new DoubleFunction() {
			@Override
			public double apply(double arg1) {
				return random.nextDouble();
			}
		});
		return randM;
	}

	/**
	 * A hard-coded initialization matrix used for validation
	 * 
	 * @param rows
	 * @param cols
	 * @return
	 */

	static Matrix randomValidationMatrix(int rows, int cols) {
		double randomArray[][] = {
				{ 0.730967787376657, 0.24053641567148587, 0.6374174253501083 },
				{ 0.5504370051176339, 0.5975452777972018, 0.3332183994766498 },
				{ 0.3851891847407185, 0.984841540199809, 0.8791825178724801 },
				{ 0.9412491794821144, 0.27495396603548483, 0.12889715087377673 },
				{ 0.14660165764651822, 0.023238122483889456, 0.5467397571984656 } };
		DenseMatrix matrix = new DenseMatrix(randomArray);
		return matrix;
	}
	
	
	/**
	   * should it pass a record during sampling
	   * @param sampleRate
	   * @return pass it or not
	*/
	static boolean pass(double sampleRate) {
		double selectionChance = random.nextDouble();
		boolean pass = (selectionChance > sampleRate);
		return pass;
	}

	
	/**
	   * @param m matrix
	   * @return trace of matrix m
	*/

	static double trace(Matrix m) {
		Vector d = m.viewDiagonal();
		return d.zSum();
	}

	/**
	 * Subtract two arrays
	 * @param res: The minuend and the variable where the difference is stored
	 * @param subtractor: the subtrahend
	 * @return difference array
	 */
	public static double[] subtract(double[] res, double[] subtractor) {
		for (int i = 0; i < res.length; i++) {
			res[i] -= subtractor[i];
		}
		return res;
	}

	/**
	 * Subtract a vector from an array
	 * @param res: The minuend and the variable where the difference is stored
	 * @param subtractor: the subtrahend
	 * @return difference array
	 */
	public static double[] subtractVectorArray(double[] res, Vector subtractor) {
		for (int i = 0; i < res.length; i++) {
			res[i] -= subtractor.getQuick(i);
		}
		return res;
	}

	/**
	 * dot product between of two arrays
	 */
	public static double dot(double[] arr1, double[] arr2) {
		double dotRes = 0;
		for (int i = 0; i < arr1.length; i++) {
			dotRes += arr1[i] * arr2[i];
		}
		return dotRes;
	}
	/**
	 * dot Product of vector and array
	 * @param vector
	 * @param arr2
	 * @return
	 */
	public static double dotVectorArray(Vector vector, double[] arr2) {
		double dotRes = 0;
		for (int i = 0; i < arr2.length; i++) {
			dotRes += vector.getQuick(i) * arr2[i];
		}
		return dotRes;
	}

	/**
	 * multiply a dense vector by a matrix
	 * @param xm_mahout: result vector
	 * @return
	 */
	static Vector denseVectorTimesMatrix(Vector vector, Matrix matrix,
			Vector xm_mahout) {
		int nRows = matrix.numRows();
		int nCols = matrix.numCols();
		for (int c = 0; c < nCols; c++) {
			double dotres = 0;
			for (int r = 0; r < nRows; r++)
				dotres += vector.getQuick(r) * matrix.getQuick(r, c);
			xm_mahout.set(c, dotres);
		}
		return xm_mahout;
	}

	/**
	 * multiply a dense vector by a matrix
	 * @param arr: result array
	 * @return
	 */
	static double[] denseVectorTimesMatrix(Vector vector, Matrix matrix,
			double[] arr) {
		int nRows = matrix.numRows();
		int nCols = matrix.numCols();
		for (int c = 0; c < nCols; c++) {
			double dotres = 0;
			for (int r = 0; r < nRows; r++)
				dotres += vector.getQuick(r) * matrix.getQuick(r, c);
			arr[c] = dotres;
		}
		return arr;
	}

	/**
	 * multiply a dense vector by the transpose of a matrix
	 * @param resArray: result array
	 * @return
	 */
	static double[] vectorTimesMatrixTranspose(Vector vector, Matrix matrix,
			double[] resArray) {
		int nRows = matrix.numRows();
		int nCols = matrix.numCols();
		for (int r = 0; r < nRows; r++) {
			double dotres = 0;
			for (int c = 0; c < nCols; c++)
				dotres += vector.getQuick(c) * matrix.getQuick(r, c);
			resArray[r] = dotres;
		}
		return resArray;
	}

	static double[] vectorTimesMatrixTranspose(double[] arr, Matrix matrix,
			double[] resArray) {
		int nRows = matrix.numRows();
		int nCols = matrix.numCols();
		for (int r = 0; r < nRows; r++) {
			double dotres = 0;
			for (int c = 0; c < nCols; c++)
				dotres += arr[c] * matrix.getQuick(r, c);
			resArray[r] = dotres;
		}
		return resArray;
	}
	static double[] vectorTimesMatrixTranspose(double[] arr, double[][] matrix,
			double[] resArray) {
		int nRows = matrix.length;
		int nCols = matrix[0].length;
		for (int r = 0; r < nRows; r++) {
			double dotres = 0;
			for (int c = 0; c < nCols; c++)
				dotres += arr[c] * matrix[r][c];
			resArray[r] = dotres;
		}
		return resArray;
	}
	
	/**
	 * Subtract a vector from a sparse vector
	 * @param sparseVector
	 * @param vector
	 * @param nonZeroIndices: the indices of nonzero elements in the sparse vector
	 * @param resArray: result array
	 * @return
	 */
	static double[] sparseVectorMinusVector(org.apache.spark.mllib.linalg.Vector sparseVector, Vector vector,
			double[] resArray, int[] nonZeroIndices) {
		int index = 0;
		double value = 0;
		for(int i=0; i<nonZeroIndices.length; i++)
		{
			index=nonZeroIndices[i];
			value=sparseVector.apply(index);
			resArray[index] = value - vector.getQuick(index); //because the array is already negated
		}
		return resArray;
	}
	/**
	 * Subtract two dense vectors
	 * @param vector1
	 * @param vector2
	 * @param resArray: result array
	 * @return
	 */
	static double[] denseVectorMinusVector(org.apache.spark.mllib.linalg.Vector vector1, Vector vector2,
			double[] resArray) {
		
		for(int i=0; i< vector2.size(); i++)
		{
			resArray[i] = vector1.apply(i) - vector2.getQuick(i);
		}
		return resArray;
	}

	/**
	 * computes the outer (tensor) product of two vectors. The result of applying the outer product on a pair of vectors is a matrix
	 * @param yi: sparse vector
	 * @param ym: mean vector
	 * @param xi: dense vector
	 * @param xm: mean vector
	 * @param nonZeroIndices:  the indices of nonzero elements in the sparse vector
	 * @param resArray: resulting two-dimensional array
	 */
	public static void outerProductWithIndices(org.apache.spark.mllib.linalg.Vector yi, Vector ym, double[] xi,
			Vector xm, double[][] resArray, int[] nonZeroIndices) {
		// 1. Sum(Yi' x (Xi-Xm))
		
		int xSize = xi.length;
		double yScale=0;
		int i, yRow;
		for(i=0; i < nonZeroIndices.length; i++)
		{
			yRow=nonZeroIndices[i];
			yScale=yi.apply(yRow);
			for (int xCol = 0; xCol < xSize; xCol++) {
				double centeredValue = xi[xCol] - xm.getQuick(xCol);
				resArray[yRow][xCol] += centeredValue * yScale;
			}
			
		}
		
	}
	
	public static void outerProductArrayInput(double[] yi, Vector ym,
			double[] xi, Vector xm, double[][] resArray) {

		int ySize = yi.length;
		int xSize = xi.length;
		for (int yRow = 0; yRow < ySize; yRow++) {
			for (int xCol = 0; xCol < xSize; xCol++) {
				double centeredValue = xi[xCol] - xm.getQuick(xCol);
				resArray[yRow][xCol] += centeredValue * yi[yRow];
			}
		}
	}

	/***
	 * Mi = (Yi-Ym)' x (Xi-Xm) = Yi' x (Xi-Xm) - Ym' x (Xi-Xm)
	 * 
	 * M = Sum(Mi) = Sum(Yi' x (Xi-Xm)) - Ym' x (Sum(Xi)-N*Xm)
	 * 
	 * The second part is done in this function
	 */
	
	public static Matrix updateXtXAndYtx(Matrix realCentralYtx,
			Vector realCentralSumX, Vector ym, Vector xm, int nRows) {
		for (int yRow = 0; yRow < ym.size(); yRow++) {
			double scale = ym.getQuick(yRow);
			for (int xCol = 0; xCol < realCentralSumX.size(); xCol++) {
				double centeredValue = realCentralSumX.getQuick(xCol) - nRows
						* xm.getQuick(xCol);
				double currValue = realCentralYtx.getQuick(yRow, xCol);
				currValue -= centeredValue * scale;
				realCentralYtx.setQuick(yRow, xCol, currValue);
			}
		}
		return realCentralYtx;
	}

	/**
	 * Subtract two arrays from one array in one loop
	 * @param mainVector: The minuend and the variable where the difference is stored 
	 * @param subtractor1: the first subtrahend
	 * @param subtractor2: the second subtrahend
	 * @return
	 */
	
	static double[] denseVectorSubtractSparseSubtractDense(double[] mainVector,
			org.apache.spark.mllib.linalg.Vector subtractor1, double[] subtractor2) {
		int nCols = mainVector.length;
		for (int c = 0; c < nCols; c++) {
			double v = mainVector[c];
			v -= subtractor1.apply(c);
			v -= subtractor2[c];
			mainVector[c] = v;
		}
		return mainVector;
	}

	/**
	 * multiply a sparse vector by a matrix
	 * @param sparseVector
	 * @param matrix
	 * @param resArray
	 */
	
	static void sparseVectorTimesMatrix(org.apache.spark.mllib.linalg.Vector sparseVector, Matrix matrix,
			double[] resArray) {
		int matrixCols = matrix.numCols();
		int[] indices;
		for (int col = 0; col < matrixCols; col++) 
		{
			indices=((SparseVector)sparseVector).indices();
			int index = 0, i=0;
			double value = 0;
			double dotRes = 0;
			for(i=0; i <indices.length; i++)
			{
				index=indices[i];
				value=sparseVector.apply(index);
				dotRes += matrix.getQuick(index,col) * value;
			}
			resArray[col] = dotRes;
		}
	}
	static org.apache.spark.mllib.linalg.Vector sparseVectorTimesMatrix(org.apache.spark.mllib.linalg.Vector sparseVector, Matrix matrix) {
		int matrixCols = matrix.numCols();
		int[] indices;
		ArrayList<Tuple2<Integer, Double>> tupleList = new  ArrayList<Tuple2<Integer, Double>>();
		for (int col = 0; col < matrixCols; col++) 
		{
			indices=((SparseVector)sparseVector).indices();
			int index = 0, i=0;
			double value = 0;
			double dotRes = 0;
			for(i=0; i <indices.length; i++)
			{
				index=indices[i];
				value=sparseVector.apply(index);
				dotRes += matrix.getQuick(index,col) * value;
			}
			if(dotRes !=0)
			{
				Tuple2<Integer,Double> tuple = new Tuple2<Integer,Double>(col,dotRes);
				tupleList.add(tuple);
			}
		}
		org.apache.spark.mllib.linalg.Vector sparkVector = Vectors.sparse(matrixCols,tupleList);
        return sparkVector;
	}
	
	/**
	 * Compute the inverse of a Matrix
	*/
	public static Matrix inv(Matrix m) {
		// assume m is square
		QRDecomposition qr = new QRDecomposition(m);
		Matrix i = eye(m.numRows());
		Matrix res = qr.solve(i);
		Matrix densRes = toDenseMatrix(res); // to go around sparse matrix bug
		return densRes;
	}
	
	/**
	 * Convert abstract matrix to dense matrix
	 */
    private static DenseMatrix toDenseMatrix(Matrix origMtx) {
		DenseMatrix mtx = new DenseMatrix(origMtx.numRows(), origMtx.numCols());
		Iterator<MatrixSlice> sliceIterator = origMtx.iterateAll();
		while (sliceIterator.hasNext()) {
			MatrixSlice slice = sliceIterator.next();
			mtx.viewRow(slice.index()).assign(slice.vector());
		}
		return mtx;
	}
    
    /**
	 * Initialize an identity matrix I
	*/
	private static Matrix eye(int n) {
		Matrix m = new DenseMatrix(n, n);
		m.assign(0);
		m.viewDiagonal().assign(1);
		return m;
	}
	
	/**
	 * get the maximum value in an array
	 */
	public static double getMax(double[] arr) {
		  double max= 0;
		  for(int i=0; i< arr.length; i++)
		  {
			  if(arr[i] > max)
				  max=arr[i];
		  }
		  return max;
	}
	
	/**
	 * Convert org.apache.mahout.math.Matrix object to org.apache.spark.mllib.linalg.Matrix object to be used in Spark Programs
	 */
	public static org.apache.spark.mllib.linalg.Matrix convertMahoutToSparkMatrix(Matrix mahoutMatrix)
	{
		int rows=mahoutMatrix.numRows();
		int cols=mahoutMatrix.numCols();
		int arraySize= rows*cols;
		int arrayIndex=0;
		double[] colMajorArray= new double[arraySize];
		for(int i=0;i<cols; i++)
		{
			for(int j=0; j< rows; j++)
			{
				colMajorArray[arrayIndex] = mahoutMatrix.get(j, i);
				arrayIndex++;
			}
		}
		org.apache.spark.mllib.linalg.Matrix sparkMatrix = Matrices.dense(rows, cols, colMajorArray);
		return sparkMatrix;
	}
	
	
	/**
	 * Writes the matrix to file based on the given format format
	 */
	public static void printMatrixToFile(org.apache.spark.mllib.linalg.Matrix m, OutputFormat format, String outputPath) {
		String outputFilePath=outputPath+ File.separator + "PCs.txt";
		switch(format)
		{
			case DENSE:
				printMatrixInDenseTextFormat(m,outputFilePath);
				break;
			case LIL:
				printMatrixInListOfListsFormat(m,outputFilePath);
				break;
			case COO:
				printMatrixInCoordinateFormat(m,outputFilePath);
				break;	
		}
	}
	
	/**
	 * Writes the matrix in a Dense text format
	 */
	
	public static void printMatrixInDenseTextFormat(org.apache.spark.mllib.linalg.Matrix m, String outputPath) {
		try {
			FileWriter fileWriter = new FileWriter(outputPath);
			PrintWriter printWriter= new PrintWriter(fileWriter);
			 for(int i=0; i < m.numRows(); i++)
			 {
				for(int j=0; j < m.numCols(); j++)
				{
					printWriter.print(m.apply(i, j) + " ");
				}
				printWriter.println();
			}
			printWriter.close();
			fileWriter.close();
		}
		catch (Exception e) {
			Log.error("Output file " + outputPath + " not found ");
		}	
	}
	
	/**
	 * Writes the matrix in a List of lists (LIL) format
	 */
	public static void printMatrixInListOfListsFormat(org.apache.spark.mllib.linalg.Matrix m, String outputPath) {
		try
		{
			FileWriter fileWriter = new FileWriter(outputPath);
			PrintWriter printWriter= new PrintWriter(fileWriter);
			boolean firstValue=true;
			double val;
			for(int i=0; i < m.numRows(); i++)
			{
				printWriter.print("{");
				for(int j=0; j < m.numCols(); j++)
				{
					val=m.apply(i, j);
					if(val!=0)
						if(firstValue)
						{
							printWriter.print(j + ":" + val);
						    firstValue=false;
						}
						else
							printWriter.print("," + j + ":" + val);
				}
				printWriter.print("}");
				printWriter.println();
				firstValue=true;
			}
			printWriter.close();
			fileWriter.close();
		}
		catch (Exception e) {
			Log.error("Output file " + outputPath + " not found ");
		}
	}
	
	/**
	 * Writes the matrix in a Coordinate list (COO) format
	 */
	public static void printMatrixInCoordinateFormat(org.apache.spark.mllib.linalg.Matrix m, String outputPath) {
		try
		{
			FileWriter fileWriter = new FileWriter(outputPath);
			PrintWriter printWriter= new PrintWriter(fileWriter);
			double val;
			for(int i=0; i < m.numRows(); i++)
			 {
				for(int j=0; j < m.numCols(); j++)
				{
					val=m.apply(i, j);
					if(val!=0)
						printWriter.println(i + "," + j + "," + val);
				}
			}
			printWriter.close();
			fileWriter.close();
		}
		catch (Exception e) {
			Log.error("Output file " + outputPath + " not found ");
		}
	}
	
}
