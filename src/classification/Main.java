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
		int fsMethod = 0;
		String clsMethod = "NB";
		int wordsToKeep = 10000;
		int numToSelect = 20;
		String outFilename = fsMethod+"-"+clsMethod+"-"+wordsToKeep+"-"+numToSelect+"-" + ".txt";
		Utility.filename = outFilename;
		Utility.ensureFileExists(outFilename);

		System.out.println("fsMethod " + fsMethod 
				+ "\nclsMethod " 
				+ clsMethod + "\nwordsToKeep " 
				+ wordsToKeep + "\nnumToSelect " 
				+ numToSelect + "\noutfile " + outFilename);

		long startTime = System.currentTimeMillis();

		DatasetHelper dh = new DatasetHelper(wordsToKeep);

		// Configure StringToWordVector using all words from the training set
		Instances originalTrainRaw = dh.loadData("alltrain_noclass.arff");
		StringToWordVector wordVectorfilter = dh.createWordVectorFilter(originalTrainRaw);
		Instances originalTrain = Filter.useFilter(originalTrainRaw, wordVectorfilter);
		List<String> possibleLabels = new ArrayList<String>(labelsUsed);
		possibleLabels.add("other");
		dh.labelDataset("alltrain_noclass_rest.arff", originalTrain, "reuters", possibleLabels);

		Utility.outputToFile("======================= LAYER 1 =======================");

		// Set up layer 1
		// ----------------------------------------------------

		Layer layer1 = null;
		// Whether to use feature selection or not.
		if(fsMethod != 0){
			layer1 = new Layer(clsMethod, dh, new FeatureSelection(fsMethod, numToSelect));
		}
		else{
			layer1 = new Layer(clsMethod, dh);
		}
		layer1.loadTrain("layer1/train/", "l1train.arff", wordVectorfilter);
		//layer1.loadTrain("temp/", wordVectorfilter);
		layer1.trainClassifiers();
		layer1.loadTest("layer1/test/l1test.arff", "layer1/test/l1test_rest.arff", wordVectorfilter);
		double[] PRFlayer1 = layer1.classify();
		
		System.out.println("Layer 1 applied to train set part 2 as test set");
		Utility.outputToFile("Layer 1 applied to train set part 2 as test set");
		System.out.println("Precision = " + PRFlayer1[0] + "\nRecall = " + PRFlayer1[1] + "\nF1 = " + PRFlayer1[2]);
		Utility.outputToFile("Precision = " + PRFlayer1[0] + "\nRecall = " + PRFlayer1[1] + "\nF1 = " + PRFlayer1[2]);

		layer1.saveClassifiersPredictions("layer2/train/", "l2train.arff", true);

		//		LibSVM svm = new LibSVM();
		//		SelectedTag kt = new SelectedTag(0, LibSVM.TAGS_KERNELTYPE);
		//		SelectedTag svmt = new SelectedTag(0, LibSVM.TAGS_SVMTYPE);
		//		svm.setKernelType(kt);
		//		svm.setSVMType(svmt);
		//		svm.setProbabilityEstimates(true);

		Utility.outputToFile("======================= LAYER 2 =======================");

		// Set up layer 2
		// ----------------------------------------------------

		Layer layer2 = new Layer(clsMethod, dh);
		layer2.loadTrain("layer2/train/", "l2train.arff", null);
		
		// Configure StringToWordVector for titles using the original training set
		originalTrainRaw.deleteAttributeAt(1);		// delete body attribute
		StringToWordVector titleWordVectorfilter = dh.createWordVectorFilter(originalTrainRaw);
		
		layer2.addMoreFeaturesToTrain(titleWordVectorfilter, "layer1/test/l1test.arff");
		
		layer2.trainClassifiers();
		
		Utility.outputToFile("======================= APPLY LAYER 1 TO TEST SET =======================");
		
		// Apply layer 1 to test set
		//
		layer1.loadTest("test_noclass.arff", "test_noclass_rest.arff", wordVectorfilter);
		double[] PRF1 = layer1.classify();
		
		System.out.println("Layer 1 applied to test set");
		Utility.outputToFile("Layer 1 applied to test set");
		System.out.println("Precision = " + PRF1[0] + "\nRecall = " + PRF1[1] + "\nF1 = " + PRF1[2]);
		Utility.outputToFile("Precision = " + PRF1[0] + "\nRecall = " + PRF1[1] + "\nF1 = " + PRF1[2]);
		
		layer1.saveClassifiersPredictions("layer2/test/", "l2test.arff", false);
		
		Utility.outputToFile("======================= APPLY LAYER 2 TO TEST SET =======================");
		
		// Apply layer 2 to test set
		//
		layer2.loadTest("layer2/test/l2test.arff", "layer2/test/l2test_rest.arff", null);
		layer2.addMoreFeaturesToTest(titleWordVectorfilter, "test_noclass.arff");
		double[] PRFlayer2 = layer2.classify();

		System.out.println("Layer 2 applied to test set");
		Utility.outputToFile("Layer 2 applied to test set");
		System.out.println("Precision = " + PRFlayer2[0] + "\nRecall = " + PRFlayer2[1] + "\nF1 = " + PRFlayer2[2]);
		Utility.outputToFile("Precision = " + PRFlayer2[0] + "\nRecall = " + PRFlayer2[1] + "\nF1 = " + PRFlayer2[2]);
		
		long endTime   = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		System.out.println("Took : " + (totalTime / 1000) + "s");
		Utility.outputToFile("Took : " + (totalTime / 1000) + "s");
	}

}
