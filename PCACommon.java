import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;


import org.apache.commons.math.linear.RealVector.Entry;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.OpenMapRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SparseRealVector;
import org.apache.commons.math3.util.Pair;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
//import org.apache.spark.mllib.linalg.Vector;
import org.apache.mahout.math.function.DoubleFunction;






/**
 * This class includes the utility functions that is used by multiple PCA
 * classes
 * 
 * @author Tarek Elgamal
 */
class PCACommon {
 // private static final Logger log = LoggerFactory.getLogger(PCACommon.class);

	  /**
	   * We use a single random object to help reproducing the erroneous scenarios
	   */
	  static Random random = new Random(0);
	  // Random random = new Random(System.currentTimeMillis());
	
	  /**
	   * @return random initialization for variance
	   */
	  static double randValidationSS() {
	    //return random.nextDouble();
	     return 0.9644868606768501;
	  }
	  
	  static double randSS() {
		 return random.nextDouble();
	  }
	
	  /**
	   * A randomly initialized matrix
	   * @param rows
	   * @param cols
	   * @return
	   */
	 
	 
	 
	 public static RealMatrix randomValidationRealMatrix(int rows, int cols) {
			/*
			RealMatrix realMatrix= MatrixUtils.createRealMatrix(rows, cols); 
				
			for (int j=0; j < cols; j++)
			  {
				    double[] array=new double[rows];
				    int i =0;
				    for (i=0; i<rows; i++)
					{
					     array[i]=0;//random.nextDouble();
					}
				    realMatrix.setColumn(j, array);
			   }
			
			   return realMatrix;
			*/
			double randomArray[][]= {{0.730967787376657, 0.24053641567148587, 0.6374174253501083},
						 {0.5504370051176339, 0.5975452777972018, 0.3332183994766498},
						 {0.3851891847407185, 0.984841540199809, 0.8791825178724801},
						 {0.9412491794821144, 0.27495396603548483, 0.12889715087377673},
						 {0.14660165764651822, 0.023238122483889456, 0.5467397571984656}};
			RealMatrix realMatrix=MatrixUtils.createRealMatrix(randomArray);
		        return realMatrix;
			
		}
	 public static RealMatrix randomRealMatrix(int rows, int cols) {
		
		RealMatrix realMatrix= MatrixUtils.createRealMatrix(rows, cols); 
			
		for (int j=0; j < cols; j++)
		  {
			    double[] array=new double[rows];
			    int i =0;
			    for (i=0; i<rows; i++)
				{
				     array[i]=random.nextDouble();
				}
			    realMatrix.setColumn(j, array);
		   }
		
		   return realMatrix;
	}
	 
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
	 
	 static Matrix randomValidationMatrix(int rows, int cols) {
		 		 double randomArray[][]= {{0.730967787376657, 0.24053641567148587, 0.6374174253501083},
				 {0.5504370051176339, 0.5975452777972018, 0.3332183994766498},
				 {0.3851891847407185, 0.984841540199809, 0.8791825178724801},
				 {0.9412491794821144, 0.27495396603548483, 0.12889715087377673},
				 {0.14660165764651822, 0.023238122483889456, 0.5467397571984656}};
		 	     DenseMatrix matrix=new DenseMatrix(randomArray);
		 		return matrix;
		  }
	
	 public static double[] convertDoubles(List<Double> doubles)
	 {
			int i=0;
		    double[] ret = new double[doubles.size()];
		    Iterator<Double> iterator = doubles.iterator();
		    while(iterator.hasNext())
		    {
		        ret[i] = iterator.next().doubleValue();
		        i++;
		    }
		    return ret;
	}
	 static double trace(Matrix m) {
		    Vector d = m.viewDiagonal();
		    return d.zSum();
		  }
	 
	 public static double[] subtract(double[] res, double[] subtractor) {
		     for(int i=0; i < res.length; i++)
		     {
		    	 res[i] -= subtractor[i];
		     }
		     return res;
		  }
	 public static double[] subtract(double[] res, Vector subtractor) {
	     for(int i=0; i < res.length; i++)
	     {
	    	 res[i] -= subtractor.getQuick(i);
	     }
	     return res;
	  }
	 public static double dot(double[] arr1, double[] arr2) {
		 double dotRes =0;
	     for(int i=0; i < arr1.length; i++)
	     {
	    	 dotRes += arr1[i] * arr2[i];
	     }
	     return dotRes;
	  }
	
	 static SparseRealVector convertSparkVectorToRealVector(org.apache.spark.mllib.linalg.Vector sparkVec) {
		 return new OpenMapRealVector(sparkVec.toArray());
		 
		 //return MatrixUtils.createRealVector(sparkVec.toArray());
		 
	 }
	 
	 public static org.apache.commons.math.linear.SparseRealVector convertExtendedRealVectorToRealVectorMath(ExtendedRealVector extendedVec) {
		 return new org.apache.commons.math.linear.OpenMapRealVector(extendedVec.toArray());
     }
	 
	 static ExtendedRealVector convertSparkVectorToExtendedRealVector(org.apache.spark.mllib.linalg.Vector sparkVec) {
		 return new ExtendedRealVector(sparkVec.toArray());
		 		 
	 }
	 static RealVector convertDenseSparkVectorToRealVector(org.apache.spark.mllib.linalg.Vector sparkVec) { 
		 return MatrixUtils.createRealVector(sparkVec.toArray());
		 
	 }
	 
	 public static org.apache.commons.math.linear.SparseRealVector convertSparkVectorToRealVectorMath(org.apache.spark.mllib.linalg.Vector sparkVec) {
		 return new org.apache.commons.math.linear.OpenMapRealVector(sparkVec.toArray());
	 }
	 
	
	public static RealMatrix addToDiagonal(RealMatrix realMatrix, double value) {
		int diagLength=realMatrix.getRowDimension(); //same as columns (square matrix
		int i;
		for(i=0; i < diagLength; i++)
		{
			double oldValue=realMatrix.getEntry(i, i);
			realMatrix.setEntry(i, i, oldValue + value);
		}
		return realMatrix;
	}

	public static RealMatrix multiplyConstantToMatrix(RealMatrix realMatrix, double value) {
		int rows=realMatrix.getRowDimension();
		int cols=realMatrix.getColumnDimension();
		int i,j;
		for(i=0; i < rows; i++)
		{
			for(j=0; j<cols; j++)
			{
				double oldValue=realMatrix.getEntry(i, j);
				realMatrix.setEntry(i, j, oldValue * value);
			}
		}
		return realMatrix;
	}
	static Vector denseVectorTimesMatrix(Vector vector, Matrix matrix,
		      DenseVector resVector) {
		    int nRows = matrix.numRows();
		    int nCols = matrix.numCols();
		    for (int c = 0; c < nCols; c++) {
		      double dotres = 0;
		      for (int r = 0; r < nRows; r++)
		        dotres += vector.getQuick(r) * matrix.getQuick(r, c);
		      resVector.set(c, dotres);
		    }
		    return resVector;
		  }
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
	 static RealVector vectorTimesMatrixTranspose(RealVector vector,
		      RealMatrix matrix) {
		    int nRows = matrix.getRowDimension();
		    int nCols = matrix.getColumnDimension();
		    RealVector resVector = MatrixUtils.createRealVector(new double[nRows]);
		    for (int r = 0; r < nRows; r++) {
		      double dotres = 0;
		      for (int c = 0; c < nCols; c++)
		        dotres += vector.getEntry(c) * matrix.getEntry(r, c);
		      resVector.setEntry(r, dotres);
		    }
		    return resVector;
		  }
	 static double[] vectorTimesMatrixTranspose(Vector vector,
		      Matrix matrix,  double[] resArray) {
		    int nRows = matrix.numRows();
		    int nCols = matrix.numCols();
		    for (int r = 0; r < nRows; r++) {
		      double dotres = 0;
		      for (int c = 0; c < nCols; c++)
		        dotres += vector.getQuick(c) * matrix.getQuick(r, c);
		      resArray[r]= dotres;
		    }
		    return resArray;
		  }
	 
	 static double[] vectorTimesMatrixTranspose(double[] arr,
		      Matrix matrix,  double[] resArray) {
		    int nRows = matrix.numRows();
		    int nCols = matrix.numCols();
		    for (int r = 0; r < nRows; r++) {
		      double dotres = 0;
		      for (int c = 0; c < nCols; c++)
		        dotres += arr[c] * matrix.getQuick(r, c);
		      resArray[r]= dotres;
		    }
		    return resArray;
		  }
	 
	 static RealVector sparseVectorTimesMatrix(ExtendedRealVector sparseVector,
		      RealMatrix matrix) {
		    int matrixRows = matrix.getRowDimension();
		    int matrixCols = matrix.getColumnDimension();
		    RealVector resVector = MatrixUtils.createRealVector(new double[matrixCols]);
		 
		    RealVector columnVector;
		    for (int col = 0; col < matrixCols; col++) {
		    	columnVector = matrix.getColumnVector(col);
		    	double dotRes=sparseVectorTimesVector(sparseVector, columnVector);
			    resVector.setEntry(col, dotRes);
		    }
		    return resVector;
		  }
	 
	 static double sparseVectorTimesVector(ExtendedRealVector sparseVector,
		      RealVector vector) {
		    
		    Iterator<Pair<Integer,Double>> pairIter= sparseVector.getSparseIterator();
		    Pair<Integer,Double> pair;
		    int index=0;
		    double value=0;
		    double dotRes=0; 
		    while(pairIter.hasNext())
		    {
		      pair=pairIter.next();	
		      index=pair.getKey();
		      value=pair.getValue();
		      dotRes+= vector.getEntry(index) * value;
		    }
		    return dotRes;
		  }
	 
	 static RealVector sparseVectorTimesMatrixMath(org.apache.commons.math.linear.RealVector sparseVector,
			 org.apache.commons.math.linear.RealMatrix matrix) {
		    int matrixRows = matrix.getRowDimension();
		    int matrixCols = matrix.getColumnDimension();
		    RealVector resVector = MatrixUtils.createRealVector(new double[matrixCols]);
		 
		    org.apache.commons.math.linear.RealVector columnVector;
		    for (int col = 0; col < matrixCols; col++) {
		    	columnVector = matrix.getColumnVector(col);
		    	double dotRes=sparseVectorTimesVectorMath(sparseVector, columnVector);
			    resVector.setEntry(col, dotRes);
		    }
		    return resVector;
		  }
	 
	 static double sparseVectorTimesVectorMath(org.apache.commons.math.linear.RealVector sparseVector,
			 org.apache.commons.math.linear.RealVector vector) {
		    
		  Iterator<Entry> iterator =  sparseVector.sparseIterator();
		  Entry	entry;
		   int index=0;
		   double value=0;
		   double dotRes=0; 
           while (iterator.hasNext()) {
		      entry=iterator.next();	
		      index=entry.getIndex();
		      value=entry.getValue();
		      dotRes+= vector.getEntry(index) * value;
		    }
		    return dotRes;
		  }
	 
	 static RealVector sparseVectorMinusVector(ExtendedRealVector sparseVector,
		      RealVector vector) {
		 RealVector resVector = MatrixUtils.createRealVector(vector.toArray());
		    Iterator<Pair<Integer,Double>> pairIter= sparseVector.getSparseIterator();
		    Pair<Integer,Double> pair;
		    int index=0;
		    double value=0;
		    double dotRes=0; 
		    while(pairIter.hasNext())
		    {
		      pair=pairIter.next();	
		      index=pair.getKey();
		      value=pair.getValue();
		      resVector.setEntry(index, value - vector.getEntry(index));
		    }
		    return resVector;
		  }
	 
	 static double[] sparseVectorMinusVector(Vector sparseVector,
		      Vector vector, double[] resArray) {
		 
		    Iterator<Vector.Element> elementIter= sparseVector.nonZeroes().iterator();
		    Vector.Element e;
		    int index=0;
		    double value=0;
		    double dotRes=0; 
		    while(elementIter.hasNext())
		    {
		      e=elementIter.next();	
		      index=e.index();
		      value=e.get();
		      resArray[index]= value - vector.getQuick(index);
		    }
		    return resArray;
		  }
	 
	 public static RealMatrix outerProduct(ExtendedRealVector yi, RealVector xi,
				RealVector xm) {
			
		 
		    Iterator<Pair<Integer,Double>> pairIter= yi.getSparseIterator();
		    Pair<Integer,Double> pair;
		    int xSize=xi.getDimension();
		    int ySize=yi.getDimension();
		    int yRow=0;
		    double yValue=0;
		    
		    RealMatrix resMatrix = MatrixUtils.createRealMatrix(ySize, xSize);
		    
		    while(pairIter.hasNext())
		    {
		    	 pair=pairIter.next();	
		    	 yRow=pair.getKey();
			     yValue=pair.getValue();
			     for (int xCol = 0; xCol < xSize; xCol++) {
			          double centeredValue = xi.getEntry(xCol) - xm.getEntry(xCol);
			          double currValue = resMatrix.getEntry(yRow, xCol);
			          currValue += centeredValue * yValue;
			          resMatrix.setEntry(yRow, xCol, currValue);
			     }
		    }
		    return resMatrix;
		}
	 
	 public static RealMatrix outerProduct(RealVector yi, RealVector xi,
				RealVector xm) {
			
		 
		    int xSize=xi.getDimension();
		    int ySize=yi.getDimension();
		    
		    
		    RealMatrix resMatrix = MatrixUtils.createRealMatrix(ySize, xSize);
		    
		   for(int yRow=0; yRow < ySize; yRow++)
		   {
			     for (int xCol = 0; xCol < xSize; xCol++) {
			          double centeredValue = xi.getEntry(xCol) - xm.getEntry(xCol);
			          double currValue = resMatrix.getEntry(yRow, xCol);
			          currValue += centeredValue * yi.getEntry(yRow);
			          resMatrix.setEntry(yRow, xCol, currValue);
			     }
		    }
		    return resMatrix;
		}
	 
	 public static RealMatrix outerProduct(RealVector yi, RealVector xi) {
			
		 
		    int xSize=xi.getDimension();
		    int ySize=yi.getDimension();
		    
		    
		    RealMatrix resMatrix = MatrixUtils.createRealMatrix(ySize, xSize);
		    
		   for(int yRow=0; yRow < ySize; yRow++)
		   {
			     for (int xCol = 0; xCol < xSize; xCol++) {
			          double currValue = resMatrix.getEntry(yRow, xCol);
			          currValue += xi.getEntry(xCol) * yi.getEntry(yRow);
			          resMatrix.setEntry(yRow, xCol, currValue);
			     }
		    }
		    return resMatrix;
		}
	 
	 public static double[][] outerProduct(Vector yi, Vector ym, double[] xi, Vector xm, double[][] resArray) {
		      // 1. Sum(Yi' x (Xi-Xm))
		      int ySize=yi.size();
		      int xSize = xi.length;
		      Iterator<Vector.Element> nonZeroElements = yi.nonZeroes().iterator();
		      while (nonZeroElements.hasNext()) {
		        Vector.Element e = nonZeroElements.next();
		        int yRow = e.index();
		        double yScale = e.get();
		        for (int xCol = 0; xCol < xSize; xCol++) {
		          double centeredValue = xi[xCol] - xm.getQuick(xCol);
		          //double currValue = resMatrix.getQuick(yRow, xCol);
		          //currValue += centeredValue * yScale;
		          //resMatrix.setQuick(yRow, xCol, currValue);
		          resArray[yRow][xCol] += centeredValue * yScale;
		          
		        }
		      }
		      return resArray;
		    }
	 
	 public static double[][] outerProductArrayInput(double[] yi, Vector ym, double[] xi, Vector xm, double[][] resArray) {
	     
	      int ySize=yi.length;
	      int xSize = xi.length;
	      for(int yRow=0; yRow < ySize; yRow++)
	      {
	    	  for (int xCol = 0; xCol < xSize; xCol++) {
		          double centeredValue = xi[xCol] - xm.getQuick(xCol);
		          //double currValue = resMatrix.getQuick(yRow, xCol);
		          //currValue += centeredValue * yScale;
		          //resMatrix.setQuick(yRow, xCol, currValue);
		          resArray[yRow][xCol] += centeredValue * yi[yRow];
		          
		        }
	      }
	      Iterator<Vector.Element> nonZeroElements = yi.nonZeroes().iterator();
	      while (nonZeroElements.hasNext()) {
	        Vector.Element e = nonZeroElements.next();
	        int yRow = e.index();
	        double yScale = e.get();
	        for (int xCol = 0; xCol < xSize; xCol++) {
	          double centeredValue = xi[xCol] - xm.getQuick(xCol);
	          //double currValue = resMatrix.getQuick(yRow, xCol);
	          //currValue += centeredValue * yScale;
	          //resMatrix.setQuick(yRow, xCol, currValue);
	          resArray[yRow][xCol] += centeredValue * yScale;
	          
	        }
	      }
	      return resArray;
	    }
	 
	 
	 static Matrix sparseVectorOuterProductVector(Vector yi, Vector ym,
			 Vector xi, Vector xm) {
		    
		    Iterator<Element> elements = yi.nonZeroes().iterator();
		    int ySize=yi.size();
		    int xSize = xi.size();
		    double[][] resArr= new double[ySize][xSize];
		    int index=0;
		    double value=0;
		    double dotRes=0;
		    Element elem;
		    while(elements.hasNext())
		    {
		    	elem=elements.next();	
		        index=elem.index();
		        value=elem.get();
		        for(int i=0; i< xi.size(); i++)
		        {
		        	resArr[index][i]= (xi.getQuick(i) - xm.getQuick(i)) * value;
		        }
		    }
		    return new DenseMatrix(resArr);
		  }

	 /***
      * Mi = (Yi-Ym)' x (Xi-Xm) = Yi' x (Xi-Xm) - Ym' x (Xi-Xm)
      * 
      * M = Sum(Mi) = Sum(Yi' x (Xi-Xm)) - Ym' x (Sum(Xi)-N*Xm)
      * 
      * The first part is done in the previous rdd transformation and the second is in this function
      */
	 public static RealMatrix updateXtXAndYtx(RealMatrix realCentralYtx , RealVector realCentralSumX , RealVector ym, RealVector xm, int nRows)
	 {
		 for (int yRow = 0; yRow < ym.getDimension(); yRow++) {
             double scale = ym.getEntry(yRow);
             for (int xCol = 0; xCol < realCentralSumX.getDimension(); xCol++) {
               double centeredValue = realCentralSumX.getEntry(xCol) - nRows
                   * xm.getEntry(xCol);
               double currValue = realCentralYtx.getEntry(yRow, xCol);
               currValue -= centeredValue * scale;
               realCentralYtx.setEntry(yRow, xCol, currValue);
             }
		 }
		 return realCentralYtx;
	 }
	 public static Matrix updateXtXAndYtx(Matrix realCentralYtx , Vector realCentralSumX , Vector ym, Vector xm, int nRows)
	 {
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

	 /*
	 static void denseVectorPlusAbsSparseVector(
		      org.apache.commons.math.linear.RealVector denseVector, org.apache.commons.math.linear.RealVector sparseVector) {
		    Iterator<org.apache.commons.math.linear.RealVector.Entry> nonZeroElements = sparseVector.sparseIterator();
		    while (nonZeroElements.hasNext()) {
		     org.apache.commons.math.linear.RealVector.Entry e = nonZeroElements.next();
		      int index = e.getIndex();
		      double v = e.getValue();
		      double prevV = denseVector.getEntry(index);
		      denseVector.setEntry(index, prevV + Math.abs(v));
		    }
		    
		  }
		  */
	 static RealVector denseVectorPlusAbsDenseDiff(
		      RealVector denseVector, RealVector sparseVector, RealVector meanVector) {
		    for (int i = 0; i < denseVector.getDimension(); i++) {
		      double denseV = denseVector.getEntry(i);
		      double v = sparseVector.getEntry(i);
		      double mean = meanVector.getEntry(i);
		      denseVector.setEntry(i, denseV + Math.abs(v-mean));
		    }
		    return denseVector;
		  }
	 static RealVector denseVectorSubtractSparseSubtractDense(RealVector mainVector,
		      ExtendedRealVector subtractor1, RealVector subtractor2) {
		    int nCols = mainVector.getDimension();
		    for (int c = 0; c < nCols; c++) {
		      double v = mainVector.getEntry(c);
		      v -= subtractor1.getEntry(c);
		      v -= subtractor2.getEntry(c);
		      mainVector.setEntry(c, v);
		    }
		    return mainVector;
		  }
	 static double[] denseVectorSubtractSparseSubtractDense(double[] mainVector,
		      Vector subtractor1, double[] subtractor2) {
		    int nCols = mainVector.length;
		    for (int c = 0; c < nCols; c++) {
		      double v = mainVector[c];
		      v -= subtractor1.getQuick(c);
		      v -= subtractor2[c];
		      mainVector[c] = v;
		    }
		    return mainVector;
		  }
	
	 static boolean pass(double sampleRate) {
		    double selectionChance = random.nextDouble();
		    boolean pass = (selectionChance > sampleRate);
		    return pass;
		  }
	 static RealVector denseVectorSubtractSparseSubtractDense(RealVector mainVector,
		      RealVector subtractor1, RealVector subtractor2) {
		    int nCols = mainVector.getDimension();
		    for (int c = 0; c < nCols; c++) {
		      double v = mainVector.getEntry(c);
		      v -= subtractor1.getEntry(c);
		      v -= subtractor2.getEntry(c);
		      mainVector.setEntry(c, v);
		    }
		    return mainVector;
		  }

	 /*
	 static Vector sparseVectorTimesMatrix(Vector vector, Matrix matrix) {
		    int nCols = matrix.numCols();
		    DenseVector resVector=new DenseVector(nCols);
		    for (int c = 0; c < nCols; c++) {
		      Double resDouble = vector.dot(matrix.viewColumn(c));
		      resVector.set(c, resDouble);
		    }
		    return resVector;
    }
	
	*/
	 static void sparseVectorTimesMatrixVoid(Vector sparseVector,
			 Matrix matrix, double[] resArray) {
		    int matrixCols = matrix.numCols();
		    
		 
		    Vector columnVector;
		    for (int col = 0; col < matrixCols; col++) {
		    	columnVector = matrix.viewColumn(col);
		    	double dotRes=sparseVectorTimesVectorManual(sparseVector, columnVector);
		    	resArray[col]=dotRes;
			    
		    }
		    
		}
	
	 
	 static Vector sparseVectorTimesMatrixManual(Vector sparseVector,
			 Matrix matrix) {
		    int matrixCols = matrix.numCols();
		    DenseVector resVector=new DenseVector(matrixCols);
		 
		    Vector columnVector;
		    for (int col = 0; col < matrixCols; col++) {
		    	columnVector = matrix.viewColumn(col);
		    	double dotRes=sparseVectorTimesVectorManual(sparseVector, columnVector);
			    resVector.set(col, dotRes);
		    }
		    return resVector;
		  }
	 
	 static double sparseVectorTimesVectorManual(Vector sparseVector,
			 Vector vector) {
		    
		 Iterator<Element> elements = sparseVector.nonZeroes().iterator();   
		 
		    int index=0;
		    double value=0;
		    double dotRes=0;
		    Element elem;
		    while(elements.hasNext())
		    {
		    	elem=elements.next();	
		        index=elem.index();
		        value=elem.get();
		        dotRes+= vector.get(index) * value;
		    }
		    return dotRes;
		  }
	 
	
	 
}


