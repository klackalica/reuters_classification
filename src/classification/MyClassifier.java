package classification;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import util.DatasetHelper;
import util.FeatureSelection;
import util.Utility;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class MyClassifier {
	
	private String clsMethod;
	private FeatureSelection fs;
	
	public MyClassifier(String clsMethod, FeatureSelection fs){
		this.clsMethod = clsMethod;
		this.fs = fs;
	}

	public Map<String, List<Double>> classify(Map<String, Instances> trainDatasets, Instances unlabeledTest, Map<String, List<Double>> testLabels){
		if(clsMethod.equals("DT")){
			return classifyDecisionTree(trainDatasets, unlabeledTest, testLabels);
		}
		else if(clsMethod.equals("NB")){
			return classifyNaiveBayes(trainDatasets, unlabeledTest, testLabels);
		}
		else{
			return classifyKNN(trainDatasets, unlabeledTest, testLabels);
		}
	}
	
	private Map<String, List<Double>> classifyDecisionTree(Map<String, Instances> trainDatasets, Instances unlabeledTest, Map<String, List<Double>> testLabels){
		Map<String, List<Double>> predictedTestLabels = new HashMap<String, List<Double>>();
		for (Map.Entry<String, Instances> entry : trainDatasets.entrySet()) {
			String labelName = entry.getKey();
			Instances labeledTest = DatasetHelper.labelDataset(testLabels.get(labelName), unlabeledTest, labelName);
			int pos = 0;
			for(int i = 0; i < labeledTest.numInstances(); i++){
				if(labeledTest.instance(i).classValue() == 1.0) pos++;
			}
			System.out.println(labelName + "\t pos = " + pos);
			System.out.println("[MyClassifier.classifyDecisionTree]\tClassifying " + labelName + "...");
			Utility.outputToFile(labelName + "\t pos = " + pos + "\n" + "[MyClassifier.classifyDecisionTree]\tClassifying " + labelName + "...");
			predictedTestLabels.put(labelName, classify(new J48(), entry.getValue(), labeledTest, labelName));
		}
		return predictedTestLabels;
	}


	private Map<String, List<Double>> classifyNaiveBayes(Map<String, Instances> trainDatasets, Instances unlabeledTest, Map<String, List<Double>> testLabels){
		Map<String, List<Double>> predictedTestLabels = new HashMap<String, List<Double>>();
		for (Map.Entry<String, Instances> entry : trainDatasets.entrySet()) {
			String labelName = entry.getKey();
			Instances labeledTest = DatasetHelper.labelDataset(testLabels.get(labelName), unlabeledTest, labelName);
			int pos = 0;
			for(int i = 0; i < labeledTest.numInstances(); i++){
				if(labeledTest.instance(i).classValue() == 1.0) pos++;
			}
			System.out.println(labelName + "\t pos = " + pos);
			System.out.println("[MyClassifier.classifyNaiveBayes]\tClassifying " + labelName + "...");
			Utility.outputToFile(labelName + "\t pos = " + pos + "\n" + "[MyClassifier.classifyNaiveBayes]\tClassifying " + labelName + "...");
			predictedTestLabels.put(labelName, classify(new NaiveBayes(), entry.getValue(), labeledTest, labelName));
		}
		return predictedTestLabels;
	}

	private Map<String, List<Double>> classifyKNN(Map<String, Instances> trainDatasets, Instances unlabeledTest, Map<String, List<Double>> testLabels){
		Map<String, List<Double>> predictedTestLabels = new HashMap<String, List<Double>>();
		for (Map.Entry<String, Instances> entry : trainDatasets.entrySet()) {
			String labelName = entry.getKey();
			Instances labeledTest = DatasetHelper.labelDataset(testLabels.get(labelName), unlabeledTest, labelName);
			int pos = 0;
			for(int i = 0; i < labeledTest.numInstances(); i++){
				if(labeledTest.instance(i).classValue() == 1.0) pos++;
			}
			System.out.println(labelName + "\t pos = " + pos);
			System.out.println("[MyClassifier.classifyKNN]\tClassifying " + labelName + "...");
			Utility.outputToFile(labelName + "\t pos = " + pos + "\n" + "[MyClassifier.classifyKNN]\tClassifying " + labelName + "...");
			predictedTestLabels.put(labelName, classify(new IBk(5), entry.getValue(), labeledTest, labelName));
		}
		return predictedTestLabels;
	}
	
	private List<Double> classify(Classifier cls, Instances train, Instances test, String labelName){
		if(fs != null){
			test = fs.filterOutAttributes(test);
			System.out.println("TEST class index = " + test.classIndex());
		}
		try {
			cls.buildClassifier(train);

			// Predict labels
			List<Double> labels = new ArrayList<Double>();
			for (int i = 0; i < test.numInstances(); i++) {
				double predicted = cls.classifyInstance(test.instance(i));
				labels.add(predicted);
			}
			return labels;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
/*
	private List<Double> classify(Classifier cls, Instances train, Instances test){
		List<Double> labels = new ArrayList<Double>();
		try {
			cls.buildClassifier(train);

			// evaluate classifier and print some statistics
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(cls, test);
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
			System.out.println(eval.toClassDetailsString("\nClass Details\n======\n"));
			System.out.println(eval.toMatrixString());
		} catch (Exception e) {
			e.printStackTrace();
		}
		return labels;
	}
*/
}
