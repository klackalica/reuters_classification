package classification;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import util.DatasetHelper;
import util.FeatureSelection;
import util.Utility;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Layer {

	public static final List<String> labelsUsed = new ArrayList<String>(
			Arrays.asList("earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "wheat",
					"ship", "corn", "money-supply", "dlr", "sugar", "oilseed", "coffee", "gnp", "gold",
					"veg-oil", "soybean", "livestock"));

	private Map<String, Instances> train = null;
	private Instances unlabeledTest = null;
	private Map<String, List<Double>> testLabels = null;
	private Map<String, List<Double>> predictedLabels = null;
	private FeatureSelection fs = null;
	private DatasetHelper dh = null;
	private MyClassifier myCls = null;
	//private Map<String, Classifier> classifiers = new HashMap<String, Classifier>(); 

	public Layer(String clsMethod, DatasetHelper dh){
		this.dh = dh;
		myCls = new MyClassifier(clsMethod);
	}

	public Layer(String clsMethod, DatasetHelper dh, FeatureSelection fs){
		this.dh = dh;
		this.fs = fs;
		myCls = new MyClassifier(clsMethod);
	}

	/**
	 * Load all training datasets and label them with their respective labels (topics/classes).
	 * 
	 * @param filter - Which filter to use to create word vector representation of each dataset
	 * @param folderPath - folder containing dataset files to be loaded
	 */
	public void loadTrain(String folderPath, String filename, StringToWordVector filter){
		Map<String, Instances> rawTrain = dh.loadAllDatasets(folderPath + filename, labelsUsed);
		if(filter != null){
			train = dh.toWordVector(rawTrain, filter);
		}
		else{
			train = rawTrain;
		}
		dh.labelAllDatasets(folderPath, train);
	}

	public void loadTest(String featuresFilepath, String labelsFilepath, StringToWordVector filter){
		// Load test dataset. So far, it's unlabelled.
		try {
			Instances rawTest = dh.loadData(featuresFilepath);
			if(filter != null){
				unlabeledTest = Filter.useFilter(rawTest, filter);
			}
			else{
				unlabeledTest = rawTest;
			}
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
	public void trainClassifiers(){
		if(fs != null){
			train = fs.selectFeatures(train);
		}
		myCls.trainClassifiers(train);
		//myCls.crossValidateAll(train);
	}



	public double[] classify(){
		predictedLabels = new HashMap<String, List<Double>>();
		for (Map.Entry<String, Instances> entry : train.entrySet()) {
			String labelName = entry.getKey();
			Instances labeledTest = dh.labelDataset(testLabels.get(labelName), unlabeledTest);
			if(fs != null){
				labeledTest = fs.filterOutAttributes(labeledTest, labelName);
			}
			int pos = 0;
			for(int i = 0; i < labeledTest.numInstances(); i++){
				if(labeledTest.instance(i).classValue() == 1.0) pos++;
			}
			System.out.println("\n" + labelName + "\t pos = " + pos);
			System.out.println("Classifying " + labelName + "...");
			predictedLabels.put(labelName, myCls.classify(labelName, labeledTest));
		}

		return Utility.calcPrecisionRecall(testLabels, predictedLabels);
	}

	public void saveClassifiersPredictions(String folder, String filename, boolean doClassValues){
		List<List<Integer>> featuresToSave = formatFeaturesToSave();

		StringBuilder attributesHeader = new StringBuilder();
		for(String labelName : labelsUsed){
			attributesHeader.append("@ATTRIBUTE " + labelName + "_class {0,1}\n");
		}

		// Features part
		Utility.ensureFileExists(folder + filename);
		StringBuilder arff = new StringBuilder();
		arff.append("@RELATION reuters\n");
		arff.append(attributesHeader.toString());
		arff.append("@DATA\n");

		String line;
		for(int i = 0; i < featuresToSave.size(); i++){
			line = featuresToSave.get(i).toString();
			arff.append(line.substring(1, line.length()-1)+"\n");
		}
		Utility.toArffFile(arff.toString(), folder + filename);

		if(doClassValues){
			// Class values part
			for(String labelName : labelsUsed){
				Utility.ensureFileExists(folder + labelName+"_rest.arff");

				StringBuilder lb = new StringBuilder();
				for(double d : testLabels.get(labelName)){
					lb.append(d + "\n");
				}
				Utility.toLabelFile(lb.toString(), folder + labelName + "_rest.arff");
			}
		}
	}
/*
	public void savePredictionsInTrainFormat(String folder, String filename){
		List<List<Integer>> featuresToSave = formatFeaturesToSave();

		StringBuilder attributesHeader = new StringBuilder();
		for(String labelName : labelsUsed){
			attributesHeader.append("@ATTRIBUTE " + labelName + "_class {0,1}\n");
		}

		// Features part
		Utility.ensureFileExists(folder + filename);
		StringBuilder arff = new StringBuilder();
		arff.append("@RELATION reuters\n");
		arff.append(attributesHeader.toString());
		arff.append("@DATA\n");

		String line;
		for(int i = 0; i < featuresToSave.size(); i++){
			line = featuresToSave.get(i).toString();
			arff.append(line.substring(1, line.length()-1)+"\n");
		}
		Utility.toArffFile(arff.toString(), folder + filename);

		// Class values part
		for(String labelName : labelsUsed){
			Utility.ensureFileExists(folder + labelName+"_rest.arff");

			StringBuilder lb = new StringBuilder();
			for(double d : testLabels.get(labelName)){
				lb.append(d + "\n");
			}
			Utility.toLabelFile(lb.toString(), folder + labelName + "_rest.arff");
		}
	}

	public void savePredictionsInTestFormat(String folder, String filename){
		List<List<Integer>> featuresToSave = formatFeaturesToSave();

		StringBuilder attributesHeader = new StringBuilder();
		for(String labelName : labelsUsed){
			attributesHeader.append("@ATTRIBUTE " + labelName + "_class {0,1}\n");
		}

		// Features part
		StringBuilder arff = new StringBuilder();
		arff.append("@RELATION reuters\n");
		arff.append(attributesHeader.toString());
		arff.append("@DATA\n");

		String line;
		for(int i = 0; i < featuresToSave.size(); i++){
			line = featuresToSave.get(i).toString();
			arff.append(line.substring(1, line.length()-1)+"\n");
		}

		Utility.toArffFile(arff.toString(), folder + filename);
	}
*/
	private List<List<Integer>> formatFeaturesToSave(){
		List<List<Integer>> featuresToSave = new ArrayList<List<Integer>>();
		// Initialise outInstances. Size is the number of instances that were in test set.
		// Each list element is a list of predicted values {0,1} for all 20 used labels.
		for(int i = 0; i < predictedLabels.get("earn").size(); i++){
			featuresToSave.add(new ArrayList<Integer>());
		}
		System.out.println("outInstances.size() " + featuresToSave.size());

		for(String labelName : labelsUsed){
			List<Double> labelPredictions = predictedLabels.get(labelName);
			for(int i = 0; i < labelPredictions.size(); i++){
				featuresToSave.get(i).add(labelPredictions.get(i).intValue());
			}
		}
		return featuresToSave;
	}

	public void addMoreFeaturesToTrain(StringToWordVector filter, String filepath) throws Exception {
		Instances data = dh.loadData(filepath);
		data.deleteAttributeAt(1);		// remove body attribute
		Instances titleData = Filter.useFilter(data, filter);
		for(Map.Entry<String, Instances> e : train.entrySet()){
			Instances copy = new Instances(titleData);
			Instances t = e.getValue();
			for(int i = 0; i < t.numAttributes(); i++){
				System.out.println("Insert attribute at " + copy.numAttributes() + " " + t.attribute(i).name());
				copy.insertAttributeAt(t.attribute(i), copy.numAttributes());
				System.out.println("After insertAt num attr is " + copy.numAttributes());
				// Add label values to the dataset.
				for (int j = 0; j < copy.numInstances(); j++) {
					double d = t.instance(j).value(i);
					copy.instance(j).setValue(copy.numAttributes()-1, d);
				}
			}
			// Set label/class index
			copy.setClassIndex(copy.numAttributes()-1);
			train.put(e.getKey(), copy);
		}
	}

	public void addMoreFeaturesToTest(StringToWordVector filter, String filepath) throws Exception {
		Instances data = dh.loadData(filepath);
		data.deleteAttributeAt(1);		// remove body attribute
		Instances titleData = Filter.useFilter(data, filter);
		//System.out.println(titleData);

		for(int i = 0; i < unlabeledTest.numAttributes(); i++){
			titleData.insertAttributeAt(unlabeledTest.attribute(i), titleData.numAttributes());

			// Add label values to the dataset.
			for (int j = 0; j < titleData.numInstances(); j++) {
				double d = unlabeledTest.instance(j).value(i);
				titleData.instance(j).setValue(titleData.numAttributes()-1, d);
			}
		}
		unlabeledTest = titleData;
		//System.out.println(unlabeledTest);
		System.out.println("num attr = " + unlabeledTest.numAttributes());
	}
}
