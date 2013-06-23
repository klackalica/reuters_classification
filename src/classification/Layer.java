package classification;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import util.DatasetHelper;
import util.FeatureSelection;
import util.Utility;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Layer {

	public static final List<String> labelsUsed = new ArrayList<String>(
			Arrays.asList("earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "wheat",
					"ship", "corn", "money-supply", "dlr", "sugar", "oilseed", "coffee", "gnp", "gold",
					"veg-oil", "soybean", "livestock"));

	private int seed = 1;
	private int folds = 10;
	private String clsMethod = "NB";
	private Map<String, Instances> train = null;
	private Instances unlabeledTest = null;
	private Instances originalTrain = null;
	private Map<String, List<Double>> testLabels = null;
	private Map<String, List<Double>> predictedLabels = null;
	private FeatureSelection fs = null;
	private DatasetHelper dh = null;
	private Map<String, Classifier> classifiers = new HashMap<String, Classifier>(); 

	public Layer(String clsMethod, DatasetHelper dh){
		this.clsMethod = clsMethod;
		this.dh = dh;
	}
	
	public Layer(String clsMethod, DatasetHelper dh, Instances originalTrain){
		this.clsMethod = clsMethod;
		this.dh = dh;
		this.originalTrain = originalTrain;
	}

	public Layer(String clsMethod, DatasetHelper dh, Instances originalTrain, FeatureSelection fs){
		this.dh = dh;
		this.fs = fs;
		this.originalTrain = originalTrain;
		this.clsMethod = clsMethod;
	}

	/**
	 * Load all training datasets and label them with their respective labels (topics/classes).
	 * 
	 * @param filter - Which filter to use to create word vector representation of each dataset
	 * @param folderPath - folder containing dataset files to be loaded
	 */
	public void loadTrain(String folderPath, StringToWordVector filter){
		Map<String, Instances> rawTrain = dh.loadAllDatasets(folderPath);
		train = dh.toWordVector(rawTrain, filter);
		dh.labelAllDatasets(folderPath, train);
	}
	
	public void loadTrain(String folderPath){
		System.out.println("Loading all datasets in " + folderPath + " folder.");
		train = dh.loadAllDatasets(folderPath);
		dh.labelAllDatasets(folderPath, train);
//		File folder = new File(folderPath);
//		File[] listOfFiles = folder.listFiles();
//		train = new HashMap<String, Instances>();
//		for (File file : listOfFiles) {
//			if (file.isFile()) {
//				Instances data = null;
//				try {
//					// Load data
//					data = dh.loadData(file.getAbsolutePath());
//				} catch (IOException e) {
//					System.out.println("[Utility.loadAllDatasets]: " + e.getMessage());
//				} catch (Exception e) {
//					System.out.println("[Utility.loadAllDatasets]: " + e.getMessage());
//				}
//				// Put (label name, unlabeledData) into the map.
//				String labelName = file.getName().split("\\.")[0];
//				data.setRelationName(labelName);
//				data.setClassIndex(data.numAttributes()-1);
//				train.put(labelName, data);	
//			}
//		}
	}

	public void loadTest(String featuresFilepath, String labelsFilepath, StringToWordVector filter){
		// Load test dataset. So far, it's unlabelled.
		try {
			unlabeledTest = Filter.useFilter(dh.loadData(featuresFilepath), filter);
			// Load true labels of the test dataset from a file and transform them into a map representation
			// where each entry is (label name, binary list of label values of each test instance).
			List<String> testLabelsFromFile = dh.loadLabelFile(labelsFilepath);
			testLabels = dh.formatTestLabels(labelsUsed, testLabelsFromFile);
		} catch (Exception e) {
			System.err.println("[Layer.loadTest] Error: " + e.getMessage());
		}
	}
	
	public void loadTest(String featuresFilepath, String labelsFilepath){
		// Load test dataset. So far, it's unlabelled.
		try {
			unlabeledTest = dh.loadData(featuresFilepath);
			// Load true labels of the test dataset from a file and transform them into a map representation
			// where each entry is (label name, binary list of label values of each test instance).
			List<String> testLabelsFromFile = dh.loadLabelFile(labelsFilepath);
			testLabels = dh.formatTestLabels(labelsUsed, testLabelsFromFile);
		} catch (Exception e) {
			System.err.println("[Layer.loadTest] Error: " + e.getMessage());
		}
	}

	/**
	 * Train binary classifier for every document label and then
	 * evaluate the performance of every classifier using cross-validation.
	 */
	public void trainAndEvaluate(){
		if(fs != null){
			train = fs.selectFeatures(train, originalTrain);
		}
		trainClassifiers();
		//crossValidateAll();
	}

	public void trainClassifiers(){
		for (Map.Entry<String, Instances> e : train.entrySet()) {
			System.out.println("Training classifier for " + e.getKey() + "...");
			Classifier cls = trainClassifier(e.getValue());
			classifiers.put(e.getKey(), cls);
		}
	}

	private Classifier trainClassifier(Instances data){
		Classifier cls = null;
		if(clsMethod.equals("DT")){
			cls = new J48();
		}
		else if(clsMethod.equals("NB")){
			cls = new NaiveBayes();
		}
		else{
			cls = new IBk(5);
		}
		try {
			cls.buildClassifier(data);
		} catch (Exception e) {
			System.err.println("[Layer.trainClassifier] Error: " + e.getMessage());
		}
		return cls;
	}

	public void crossValidateAll(){
		for (Map.Entry<String, Instances> e : train.entrySet()) {
			System.out.println("Cross validate " + e.getKey() + "...");
			
			// randomize data
			Random rand = new Random(seed);
			Instances randData = new Instances(e.getValue());
			randData.randomize(rand);
			if (randData.classAttribute().isNominal()){
				randData.stratify(folds);
			}
			crossValidateData(randData, e.getKey());
		}
	}

	private void crossValidateData(Instances data, String labelName){
		StringBuilder sb = new StringBuilder();
		sb.append("\n=== " + labelName + " ===\n");
		sb.append("Dataset: " + data.relationName() + "\n");
		sb.append("Seed: " + seed + "\n");
		sb.append("Folds: " + folds + "\n\n");

		Evaluation evalAll = null;
		try {
			evalAll = new Evaluation(data);
			for (int n = 0; n < folds; n++) {
				Instances train = data.trainCV(folds, n);
				Instances test = data.testCV(folds, n);

				// build and evaluate classifier
				Classifier cls = trainClassifier(train);
				evalAll.evaluateModel(cls, test);
			}

			sb.append(evalAll.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));
			//sb.append(evalAll.toMatrixString("Confusion Matrix"));
			sb.append("\n");

			System.out.println(sb.toString());
			Utility.outputToFile(sb.toString());
		} catch (Exception e) {
			System.err.println("[Layer.crossValidateData] Error: " + e.getMessage());
		}
	}

	public double[] classify(){
		MyClassifier myCls = new MyClassifier(clsMethod);
		
		predictedLabels = new HashMap<String, List<Double>>();
		for (Map.Entry<String, Instances> entry : train.entrySet()) {
			String labelName = entry.getKey();
			Instances labeledTest = dh.labelDataset(testLabels.get(labelName), unlabeledTest, labelName);
			if(fs != null){
				labeledTest = fs.filterOutAttributes(labeledTest);
			}
			int pos = 0;
			for(int i = 0; i < labeledTest.numInstances(); i++){
				if(labeledTest.instance(i).classValue() == 1.0) pos++;
			}
			System.out.println("\n" + labelName + "\t pos = " + pos);
			System.out.println("Classifying " + labelName + "...");
			Utility.outputToFile("\n" + labelName + "\t pos = " + pos + "\n" + "Classifying " + labelName + "...");
			predictedLabels.put(labelName, myCls.classify(classifiers.get(labelName), labeledTest));
		}
		
		return Utility.calcPrecisionRecall(testLabels, predictedLabels);
	}
	
	public void savePredictionsInTrainFormat(String folder){
		List<List<Integer>> outInstances = new ArrayList<List<Integer>>();
		// Initialise outInstances. Size is the number of instances that were in test set.
		// Each list element is a list of predicted values {0,1} for all 20 used labels.
		for(int i = 0; i < predictedLabels.get("earn").size(); i++){
			outInstances.add(new ArrayList<Integer>());
		}
		System.out.println("outInstances.size() " + outInstances.size());
	
		for(String labelName : labelsUsed){
			List<Double> labelPredictions = predictedLabels.get(labelName);
			for(int i = 0; i < labelPredictions.size(); i++){
				outInstances.get(i).add(labelPredictions.get(i).intValue());
			}
		}
		
		// Now label outInstances with true class values of all 20 labels
		// and write all 20 datasets to 20 different files.
		// These files serve as training data for layer 2.
		StringBuilder attributesHeader = new StringBuilder();
		for(String labelName : labelsUsed){
			attributesHeader.append("@ATTRIBUTE " + labelName + " {0,1}\n");
		}
		
		for(String labelName : labelsUsed){
			System.out.println("Building arff for " + labelName);
			StringBuilder arff = new StringBuilder();
			arff.append("@RELATION " + labelName + "\n");
			arff.append(attributesHeader.toString());
			//arff.append("@ATTRIBUTE " + labelName + "_label {0,1}\n");
			arff.append("@DATA\n");
			
			List<Double> trueLabelValues = testLabels.get(labelName);
			String line;
			for(int i = 0; i < trueLabelValues.size(); i++){
				line = outInstances.get(i).toString();
				//arff.append(line.substring(1, line.length()-1) + ", " + trueLabelValues.get(i).intValue() + "\n");
				arff.append(line.substring(1, line.length()-1) + "\n");
			}
			//System.out.println(arff.toString());
			Utility.ensureFileExists(folder + labelName+".arff");
			Utility.ensureFileExists(folder + labelName+"_rest.arff");
			Utility.toArffFile(arff.toString(), folder + labelName+".arff");
			
			StringBuilder lb = new StringBuilder();
			for(double d : trueLabelValues){
				lb.append(d + "\n");
			}
			Utility.toLabelFile(lb.toString(), folder + labelName+"_rest.arff");
		}
	}
	
	public void savePredictionsInTestFormat(String folder){
		List<List<Integer>> outInstances = new ArrayList<List<Integer>>();
		// Initialise outInstances. Size is the number of instances that were in test set.
		// Each list element is a list of predicted values {0,1} for all 20 used labels.
		for(int i = 0; i < predictedLabels.get("earn").size(); i++){
			outInstances.add(new ArrayList<Integer>());
		}
		System.out.println("outInstances.size() " + outInstances.size());
	
		for(String labelName : labelsUsed){
			List<Double> labelPredictions = predictedLabels.get(labelName);
			for(int i = 0; i < labelPredictions.size(); i++){
				outInstances.get(i).add(labelPredictions.get(i).intValue());
			}
		}

		StringBuilder attributesHeader = new StringBuilder();
		for(String labelName : labelsUsed){
			attributesHeader.append("@ATTRIBUTE " + labelName + " {0,1}\n");
		}
		
		// Features part
		StringBuilder arff = new StringBuilder();
		arff.append("@RELATION reuters\n");
		arff.append(attributesHeader.toString());
		//arff.append("@ATTRIBUTE " + labelName + "_label {0,1}\n");
		arff.append("@DATA\n");
		
		String line;
		for(int i = 0; i < outInstances.size(); i++){
			line = outInstances.get(i).toString();
			arff.append(line.substring(1, line.length()-1)+"\n");
		}
		
		Utility.toArffFile(arff.toString(), folder + "l2test.arff");
	}
}
