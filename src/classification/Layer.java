package classification;

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
		train = dh.loadAllDatasets(folderPath, filter);
		dh.labelAllDatasets(folderPath, train);
		for(Instances i : train.values()){
			System.out.println(i.relationName() + ": " + i.numAttributes());
		}
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

	/**
	 * Train binary classifier for every document label and then
	 * evaluate the performance of every classifier using cross-validation.
	 */
	public void trainAndEvaluate(){
		if(fs != null){
			train = fs.selectFeatures(train, originalTrain);
		}
		trainClassifiers();
		crossValidateAll();
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
			predictedLabels.put(labelName, myCls.classify(entry.getValue(), labeledTest));
		}
		
		return Utility.calcPrecisionRecall(testLabels, predictedLabels);
	}
}
