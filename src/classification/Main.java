package classification;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
		//
		Instances all_train = Utility.loadData("alltrain_noclass.arff");
		StringToWordVector filter = new StringToWordVector();
		filter.setOptions(new String[]{"-R first-last", "-W 3000", "-prune-rate -1.0", "-I", "-N 0"});
		filter.setWordsToKeep(2000);
		filter.setIDFTransform(true);
		filter.setInputFormat(all_train);

		// Load all training files. The instances in these files are not labelled.
		//
		File folder = new File("temp/");
		File[] listOfFiles = folder.listFiles();
		Map<String, Instances> trainDatasets = new HashMap<String, Instances>();
		for (File file : listOfFiles) {
			if (file.isFile() && !file.getName().endsWith("_rest.arff")) {
				//System.out.println("Loading " + file.getAbsolutePath());
				Instances unlabeledData = Filter.useFilter(
						Utility.loadData(file.getAbsolutePath()), filter);			// Load data and convert to word vector.
				trainDatasets.put(file.getName().split("\\.")[0], unlabeledData);	// Put (topic name, unlabeledData) into the map.
				//System.out.println(unlabeledData);
			}
		}

		// Add label/class column to training sets along with a label value of each instance.
		//
		for (File file : listOfFiles) {
			if (file.isFile() && file.getName().endsWith("_rest.arff")) {
				String labelName =  file.getName().split("_")[0];
				Utility.labelDataset(file.getAbsolutePath(), trainDatasets.get(labelName), labelName);
			}
		}

		// Load test dataset. So far, it's unlabelled.
		//
		Instances unlabeledTest = Filter.useFilter(Utility.loadData("test_noclass.arff"), filter);

		// Load real labels of the test dataset from a file and transform them into a map representation
		// where each entry is (label name, binary list of label values of each test instance).
		//
		List<String> testLabelsFromFile = Utility.loadLabelFile("test_noclass_rest.arff");
		Map<String, List<Double>> testLabels = Utility.formatTestLabels(labelsUsed, testLabelsFromFile);
		

		MyClassifier myClassifier = new MyClassifier();
		myClassifier.classifyDecisionTree(trainDatasets, unlabeledTest, testLabels);

		// Perform feature selection
		//		FeatureSelection fs = new FeatureSelection();
		//		List<Instances> fstrainDatasets = new ArrayList<Instances>();
		//		for(Instances inst : trainDatasets){
		//			fstrainDatasets.add(fs.GainRatioRanker(inst));
		//		}


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
