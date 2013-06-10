package classification;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import util.Utility;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class MyClassifier {
	
	private List<String> usedLabels = new ArrayList<String>(
			Arrays.asList("earn","acq","money-fx","grain","crude","trade","interest",
					"wheat","ship","corn","money-supply","dlr","sugar","oilseed",
					"coffee","gnp","gold","veg-oil","soybean","livestock"));

	public void classifyDecisionTree(Map<String, Instances> trainDatasets, Instances unlabeledTest){
		// Read in true labels of the test dataset
		List<String> trueLabels = Utility.loadClassValues("test_noclass_rest.arff");
		Map<String, List<Double>> predictedLabels = new HashMap<String, List<Double>>();
		for (Map.Entry<String, Instances> entry : trainDatasets.entrySet()) {
			Instances emptyLabeledTest = Utility.addClassToTestDataset(unlabeledTest, entry.getKey());
			System.out.println("Classify " + entry.getKey());
			predictedLabels.put(entry.getKey(), classify(new J48(), entry.getValue(), emptyLabeledTest));
		}
		correctness(trueLabels, predictedLabels);
	}

	private void correctness(List<String> trueLabels, Map<String, List<Double>> predictedLabels) {
		// predictedLabels: label -> list of 1's and 0's with length == #instances
		// trueLabels: list of strings "not_earn,not_acq,not_money-fx,not_grain..." with length == #instances
		long correct = 0;
		long total = 0;
		for (Map.Entry<String, List<Double>> e : predictedLabels.entrySet()) {
			//System.out.println("classname: " + e.getKey());
			for(String tlabel : trueLabels){
				for(Double plabel : e.getValue()){
					if(usedLabels.contains(e.getKey())){
						if(tlabel.contains("not_"+e.getKey())){
							correct += plabel == 0.0? 1 : 0;
						}
						else{
							correct += plabel == 1.0? 1 : 0;
						}
					}
					else{
						correct += plabel == 0.0? 1 : 0;
					}
					total++;
				}
			}
		}
		System.out.println("Correct = " + correct);
		System.out.println("Total = " + total);
	}

	public void classifyNaiveBayes(Map<String, Instances> trainDatasets, Instances unlabeledTest){
		// Read in test class values (labels/topics)
		List<String> classValues = Utility.loadClassValues("test_noclass_rest.arff");
		for (Map.Entry<String, Instances> entry : trainDatasets.entrySet()) {
			Instances labeledTest = Utility.addClassToTestDataset(classValues, unlabeledTest, entry.getKey());
			System.out.println(labeledTest);
			classify(new NaiveBayes(), entry.getValue(), labeledTest);
		}
	}

	public void classifyKNN(Map<String, Instances> trainDatasets, Instances unlabeledTest){
		// Read in test class values (labels/topics)
		List<String> classValues = Utility.loadClassValues("test_noclass_rest.arff");
		for (Map.Entry<String, Instances> entry : trainDatasets.entrySet()) {
			Instances labeledTest = Utility.addClassToTestDataset(classValues, unlabeledTest, entry.getKey());
			System.out.println(labeledTest);
			classify(new IBk(5), entry.getValue(), labeledTest);
		}
	}

	private List<Double> classify(Classifier cls, Instances train, Instances test){
		List<Double> labels = new ArrayList<Double>();
		try {
			cls.buildClassifier(train);

			// label instances
			for (int i = 0; i < test.numInstances(); i++) {
				labels.add(cls.classifyInstance(test.instance(i)));
				//test.instance(i).setClassValue(clsLabel);
			}

			// evaluate classifier and print some statistics
			//			Evaluation eval = new Evaluation(train);
			//			eval.evaluateModel(cls, test);
			//			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
			//System.out.println("Error rate: " + eval.errorRate());
			//System.out.println("PCt correct: " + eval.pctCorrect());
		} catch (Exception e) {
			e.printStackTrace();
		}
		return labels;
	}

}
