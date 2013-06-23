package util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;

public class Utility {

	public static String filename;

	public static double[] calcPrecisionRecall(Map<String, List<Double>> trueLabels, Map<String, List<Double>> predictedLabels){
		int numInstances = trueLabels.get("earn").size();
		double sumP = 0;
		double sumR = 0;
		double sumF = 0;

		for(int i = 0; i < numInstances; i++){
			int correctlyPredicted = 0;
			double numPredicted = 0;
			double numActual = 0;
			double trueLab = 0;
			double predictedLab = 0;
			String labelName = null;

			for(Map.Entry<String, List<Double>> e : trueLabels.entrySet()){
				labelName = e.getKey();
				trueLab = e.getValue().get(i);		// true label value of instance i for labelName
				predictedLab = predictedLabels.get(labelName).get(i);
				if(trueLab == 1.0){
					numActual++;
					if(trueLab == predictedLab){
						correctlyPredicted++;
					}
				}
				if(predictedLab == 1.0){
					numPredicted++;
				}
			}
			if(numActual != 0){
				sumP += correctlyPredicted / numActual;
			}
			if(numPredicted != 0){
				sumR += correctlyPredicted / numPredicted;
			}
			if((numActual + numPredicted) != 0){ 
				sumF += (2*correctlyPredicted)/(numActual + numPredicted);
			}
		}
		return new double[]{sumP/numInstances, sumR/numInstances, sumF/numInstances};
	}

	public static void outputToFile(String text){
		try {
			PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(filename, true)));
			out.println(text);
			out.close();
		} catch (IOException e) {
			System.err.println("[Utility.outputToFile]: " + e.getMessage());
		}
	}
	
	/**
	 * Ensure that the output file exist.
	 */
	public static void ensureFileExists() {
		File file = new File(filename);
		if (!file.exists()) {
			try {
				file.createNewFile();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
}
