/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.qcri.sparkpca;

import org.apache.spark.AccumulatorParam;

/**
 * This class supports Accumulator of type double[][]. It implements an element-by-element add operation for
 * two double[][]
 * 
 * @author Tarek Elgamal
 *
 */

public class MatrixAccumulatorParam implements AccumulatorParam<double[][]> {


	public double[][] addInPlace(double[][] arg0, double[][] arg1) {
		int i,j;
		int rows=arg0.length;
		int cols=arg0[0].length;
		//double[][] res = new double[rows][cols];
		for(i=0; i< rows; i++)
		{
			for(j=0; j<cols; j++)
			{
				arg0[i][j] = arg0[i][j] + arg1[i][j];
			}
		}
		return arg0;
	}

	public double[][] zero(double[][] arg0) {
		return arg0;
	}

	public double[][] addAccumulator(double[][] arg0, double[][] arg1) {
		return addInPlace(arg0,arg1);
	}
	

}
