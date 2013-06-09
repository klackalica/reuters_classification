package classification;

import java.io.File;
import java.util.Map;

import util.Utility;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;


public class Main {

	public static void main(String[] args) throws Exception {
		long startTime = System.currentTimeMillis();

		Instances all_train = Utility.loadData("alltrain_noclass.arff");	// Used to configure the stringToWordVector filter
		StringToWordVector filter = new StringToWordVector();
		filter.setOptions(new String[]{"-R first-last", "-W 3000", "-prune-rate -1.0", "-I", "-N 0"});
		filter.setWordsToKeep(2000);
		filter.setIDFTransform(true);
		filter.setInputFormat(all_train);  	// initializing the filter once with training set

		// Load all training files
		File folder = new File("temp/");
		File[] listOfFiles = folder.listFiles();
		Map<String, Instances> trainDatasets = Utility.loadAllTrainDatasets(listOfFiles, filter);

		// Add class values to all train datasets
		Utility.addClassToAllTrainDatasets(listOfFiles, trainDatasets);

		Instances iTest = Utility.loadData("test_noclass.arff");
		Instances unlabeledTest = Filter.useFilter(iTest, filter);
		
		MyClassifier myClassifier = new MyClassifier();
		myClassifier.classifyDecisionTree(trainDatasets, unlabeledTest);
		
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
