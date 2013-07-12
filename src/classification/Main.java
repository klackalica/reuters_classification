package classification;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import util.AugmentInput;
import util.AugmentL1Features;
import util.AugmentTitle;
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
		int fsMethod = 3; // 0 no feature selection 3 - gain ration ranker search
		int augmentMethod = 1;
		String clsMethod = "NB";
		int wordsToKeep = 10000;
		int numToSelect = 20; // with fsMethod 3
		int minTermFreq = 1;
		int nGrams = 1; // set to 1 to not use nGrams
		boolean useStopList = false;
		boolean useStemmer = false;
		boolean normalizeDocLength = false;
		
		String outFilename = fsMethod+"-"+clsMethod+"-"+wordsToKeep+"-"+numToSelect+"-" +augmentMethod
				+ minTermFreq+"-stopList-"+useStopList +"-"+nGrams+"Grams"+ "-stemmer-"+ useStemmer +"-normalizeDocLength-" + normalizeDocLength +".txt";
		
		Utility.filename = outFilename;
		Utility.ensureFileExists(outFilename);

		Utility.outputDual("fsMethod " + fsMethod 
				+ "\nclsMethod " + clsMethod 
				+ "\nwordsToKeep " 	+ wordsToKeep 
				+ "\nnumToSelect " + numToSelect 
				+ "\nminTermFreq " + minTermFreq
				+ "\nuseStopList "+ useStopList
				+ "\nnGarms "+ nGrams
				+ "\nuseStemmer "+ useStemmer
				+ "\nnormalizeDocLength "+ normalizeDocLength
				+ "\noutfile " + outFilename);

		long startTime = System.currentTimeMillis();

		DatasetHelper dh = new DatasetHelper(wordsToKeep, minTermFreq, useStopList, nGrams, useStemmer, normalizeDocLength);

		// Configure StringToWordVector using all words from the training set
		Instances originalTrainRaw = dh.loadData("alltrain_noclass.arff");
		StringToWordVector wordVectorfilter = dh.createWordVectorFilter(originalTrainRaw);
		Instances originalTrain = Filter.useFilter(originalTrainRaw, wordVectorfilter);
		List<String> possibleLabels = new ArrayList<String>(labelsUsed);
		possibleLabels.add("other");
		dh.labelDataset("alltrain_noclass_rest.arff", originalTrain, "reuters", possibleLabels);

		// Set up layer 1
		// ----------------------------------------------------
		Utility.outputDual("======================= LAYER 1 =======================");

		Layer layer1 = new Layer(clsMethod, dh);
		if(fsMethod != 0){
			layer1.setFeatureSelection(new FeatureSelection(fsMethod, numToSelect));
		}
		layer1.loadTrain("layer1/train/", "l1train.arff", wordVectorfilter);
		layer1.performFeatureSelectionPerLabel();
		layer1.trainClassifiers();
		layer1.loadTest("layer1/test/l1test.arff", "layer1/test/l1test_rest.arff", wordVectorfilter);
		double[] PRFlayer1 = layer1.classify();

		Utility.outputDual("Layer 1 applied to train set part 2 as test set");
		Utility.outputDual("Precision = " + PRFlayer1[0] + "\nRecall = " + PRFlayer1[1] + "\nF1 = " + PRFlayer1[2]);
		
		// Save output of layer 1. This output is used to train layer 2.
		layer1.saveClassifiersPredictions("layer2/train/", "l2train.arff", true);

		// Configure StringToWordVector for titles using the original training set
		originalTrainRaw.deleteAttributeAt(1);		// delete body attribute
		StringToWordVector titleWordVectorfilter = dh.createWordVectorFilter(originalTrainRaw);

		// Set up layer 2
		// ----------------------------------------------------
		Utility.outputDual("======================= LAYER 2 =======================");

		Layer layer2 = new Layer(clsMethod, dh);
		layer2.loadTrain("layer2/train/", "l2train.arff", null);
		AugmentInput augmentInput1 = null;
		if(augmentMethod == 1){
			augmentInput1 = new AugmentTitle("layer1/test/l1test.arff", dh, titleWordVectorfilter);
		}
		else if (augmentMethod == 2){
			augmentInput1 = new AugmentL1Features(layer1.getTestDataset());
		}
		layer2.augmentTrain(augmentInput1);	
		layer2.trainClassifiers();

		// Apply layer 1 to test set
		// ----------------------------------------------------
		Utility.outputDual("======================= APPLY LAYER 1 TO TEST SET =======================");

		layer1.loadTest("test_noclass.arff", "test_noclass_rest.arff", wordVectorfilter);
		double[] PRF1 = layer1.classify();

		Utility.outputDual("Layer 1 applied to test set");
		Utility.outputDual("Precision = " + PRF1[0] + "\nRecall = " + PRF1[1] + "\nF1 = " + PRF1[2]);

		layer1.saveClassifiersPredictions("layer2/test/", "l2test.arff", false);

		Utility.outputToFile("======================= APPLY LAYER 2 TO TEST SET =======================");

		// Apply layer 2 to test set
		// ----------------------------------------------------
		AugmentInput augmentInput2 = null;
		if(augmentMethod == 1){
			augmentInput2 = new AugmentTitle("test_noclass.arff", dh, titleWordVectorfilter);
		}
		else if (augmentMethod == 2){
			augmentInput2 = new AugmentL1Features(layer1.getTestDataset());
		}
		layer2.loadTest("layer2/test/l2test.arff", "layer2/test/l2test_rest.arff", null);		
		layer2.augmentTest(augmentInput2);
		double[] PRFlayer2 = layer2.classify();

		Utility.outputDual("Layer 2 applied to test set");
		Utility.outputDual("Precision = " + PRFlayer2[0] + "\nRecall = " + PRFlayer2[1] + "\nF1 = " + PRFlayer2[2]);

		long endTime   = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		Utility.outputDual("Took : " + (totalTime / 1000) + "s");
	}

}
