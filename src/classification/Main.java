package classification;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import util.DatasetHelper;
import util.FeatureSelection;
import util.Utility;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Main {

	public static final List<String> labelsUsed = new ArrayList<String>(
			Arrays.asList("earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "wheat",
					"ship", "corn", "money-supply", "dlr", "sugar", "oilseed", "coffee", "gnp", "gold",
					"veg-oil", "soybean", "livestock"));

	public static void main(String[] args) throws Exception {
//		int fsMethod = Integer.parseInt(args[0]);
//		String clsMethod = args[1];
//		int wordsToKeep = Integer.parseInt(args[2]);
//		int numToSelect = Integer.parseInt(args[3]);
//		Utility.filename = args[4];
		int fsMethod = 3;
		String clsMethod = "NB";
		int wordsToKeep = 5000;
		int numToSelect = 400;
		Utility.filename = "test.txt";
		Utility.ensureFileExists();
		
		System.out.println("fsMethod " + fsMethod 
				+ "\nclsMethod " 
				+ clsMethod + "\nwordsToKeep " 
				+ wordsToKeep + "\nnumToSelect " 
				+ numToSelect + "\noutfile " + "args[4]");
		
		long startTime = System.currentTimeMillis();
		
		DatasetHelper dh = new DatasetHelper(wordsToKeep);
		
		// Configure StringToWordVector using all words from the training set
		Instances originalTrainRaw = dh.loadData("alltrain_noclass.arff");
		StringToWordVector wordVectorfilter = dh.createWordVectorFilter(originalTrainRaw);
		Instances originalTrain = Filter.useFilter(originalTrainRaw, wordVectorfilter);
		List<String> possibleLabels = new ArrayList<String>(labelsUsed);
		possibleLabels.add("other");
		dh.labelDataset("alltrain_noclass_rest.arff", originalTrain, "reuters", possibleLabels);
			
//		Instances rawTrain = dh.loadData("alltrain_class.arff");
//		StringToWordVector filter = dh.createWordVectorFilter(rawTrain);
//		Instances originalTrain = Filter.useFilter(rawTrain, filter);
//		originalTrain.setClassIndex(0);
		
		Utility.outputToFile("======================= LAYER 1 =======================");
		
		// Set up layer 1
		// ----------------------------------------------------
		
		Layer layer1 = null;
		// Whether to use feature selection or not.
		if(fsMethod != 0){
			layer1 = new Layer(clsMethod, dh, originalTrain, new FeatureSelection(fsMethod, numToSelect));
		}
		else{
			layer1 = new Layer(clsMethod, dh, originalTrain);
		}
		layer1.loadTrain("layer1/train/", wordVectorfilter);
		//layer1.loadTrain("temp/", wordVectorfilter);
		layer1.trainAndEvaluate();
		layer1.loadTest("layer1/test/l1test.arff", "layer1/test/l1test_rest.arff", wordVectorfilter);
		double[] PRF = layer1.classify();
		
		System.out.println("Precision = " + PRF[0] + "\nRecall = " + PRF[1] + "\nF1 = " + PRF[2]);
		Utility.outputToFile("Precision = " + PRF[0] + "\nRecall = " + PRF[1] + "\nF1 = " + PRF[2]);

		//		LibSVM svm = new LibSVM();
		//		SelectedTag kt = new SelectedTag(0, LibSVM.TAGS_KERNELTYPE);
		//		SelectedTag svmt = new SelectedTag(0, LibSVM.TAGS_SVMTYPE);
		//		svm.setKernelType(kt);
		//		svm.setSVMType(svmt);
		//		svm.setProbabilityEstimates(true);		

		long endTime   = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		System.out.println("Took : " + (totalTime / 1000) + "s");
		Utility.outputToFile("Took : " + (totalTime / 1000) + "s");
	}

}
