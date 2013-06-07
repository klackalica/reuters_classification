package classification;

import util.DatasetIO;
import util.FeatureSelection;
import weka.core.Instances;


public class Main {

	public static void main(String[] args) throws Exception {
		long startTime = System.currentTimeMillis();
		
		Instances dataset = (new DatasetIO()).loadData();
		Instances reduced_dataset = (new FeatureSelection()).GainRatioRanker(dataset);
		System.out.println(reduced_dataset.toString());
		
		long endTime   = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		System.out.println("Took : " + (totalTime / 1000) + "s");
	}

}
