package classification;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

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
		long startTime = System.currentTimeMillis();

		// Configure StringToWordVector using all words from the training set
		Instances all_train = Utility.loadData("alltrain_noclass.arff");
		StringToWordVector filter = new StringToWordVector();
		filter.setOptions(new String[]{"-R first-last", "-W 3000", "-prune-rate -1.0", "-I", "-N 0"});
		filter.setWordsToKeep(2000);
		filter.setIDFTransform(true);
		filter.setInputFormat(all_train);

		// Load all training files. The instances in these files are not labelled.
		Map<String, Instances> trainDatasets = Utility.loadAllDatasets("temp/", filter);

		// Add label/class column to training sets along with a label value for each instance.
		Utility.labelAllDatasets("temp/", trainDatasets);
		
		// Perform feature selection
//		FeatureSelection fs = new FeatureSelection();
//		Map<String, Instances> fstrainDatasets = new HashMap<String, Instances>();
//		for(Map.Entry<String, Instances> e : trainDatasets.entrySet()){
//			fstrainDatasets.put(e.getKey(), fs.GainRatioRanker(e.getValue()));
//		}

		// Load test dataset. So far, it's unlabelled.
		Instances unlabeledTest = Filter.useFilter(Utility.loadData("test_noclass.arff"), filter);

		// Load true labels of the test dataset from a file and transform them into a map representation
		// where each entry is (label name, binary list of label values of each test instance).
		List<String> testLabelsFromFile = Utility.loadLabelFile("test_noclass_rest.arff");
		Map<String, List<Double>> testLabels = Utility.formatTestLabels(labelsUsed, testLabelsFromFile);

		MyClassifier myClassifier = new MyClassifier();
		Map<String, List<Double>> predictedLabels = myClassifier.classifyDecisionTree(trainDatasets, unlabeledTest, testLabels);
		//Map<String, List<Double>> predictedLabels = myClassifier.classifyDecisionTree(fstrainDatasets, unlabeledTest, testLabels);
		
		double[] PR = Utility.calcPrecisionRecall(testLabels, predictedLabels);
		System.out.println("Precision = " + PR[0] + "\nRecall = " + PR[1]);

		//		LibSVM svm = new LibSVM();
		//		SelectedTag kt = new SelectedTag(0, LibSVM.TAGS_KERNELTYPE);
		//		SelectedTag svmt = new SelectedTag(0, LibSVM.TAGS_SVMTYPE);
		//		svm.setKernelType(kt);
		//		svm.setSVMType(svmt);
		//		svm.setProbabilityEstimates(true);		

		long endTime   = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		System.out.println("Took : " + (totalTime / 1000) + "s");
	}

}
