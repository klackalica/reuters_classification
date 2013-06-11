package classification;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import util.Utility;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class MyClassifier {

	public void classifyDecisionTree(Map<String, Instances> trainDatasets, Instances unlabeledTest, Map<String, List<Double>> testLabels){
		for (Map.Entry<String, Instances> entry : trainDatasets.entrySet()) {
			String labelName = entry.getKey();
			Instances labeledTest = Utility.labelDataset(testLabels.get(labelName), unlabeledTest, labelName);
			int pos = 0;
			for(int i = 0; i < labeledTest.numInstances(); i++){
				if(labeledTest.instance(i).classValue() == 1.0) pos++;
			}
			//System.out.println(labeledTest);
			System.out.println("STAT for " + labelName + "\t pos = " + pos);
			classify(new J48(), entry.getValue(), labeledTest);
		}
	}

	private void correctness(List<String> trueLabels, Map<String, List<Double>> predictedLabels) {
		// predictedLabels: label -> list of 1's and 0's with length == #instances
		// trueLabels: list of strings "not_earn,not_acq,not_money-fx,not_grain..." with length == #instances
		long correct = 0;
		long total = 0;
		for (Map.Entry<String, List<Double>> e : predictedLabels.entrySet()) {
			for(String tlabel : trueLabels){
				for(Double plabel : e.getValue()){
					if(tlabel.contains("not_"+e.getKey())){
						correct += plabel == 0.0? 1 : 0;
					}
					else{
						correct += plabel == 1.0? 1 : 0;
					}
					total++;
				}
			}
		}
		System.out.println("Correct = " + correct);
		System.out.println("Total = " + total);
	}

	public void classifyNaiveBayes(Map<String, Instances> trainDatasets, Instances unlabeledTest, Map<String, List<Double>> testLabels){
		// Read in test class values (labels/topics)
		List<String> classValues = Utility.loadLabelFile("test_noclass_rest.arff");
		for (Map.Entry<String, Instances> entry : trainDatasets.entrySet()) {
			String labelName = entry.getKey();
			Instances labeledTest = Utility.labelDataset(testLabels.get(labelName), unlabeledTest, labelName);
			System.out.println(labeledTest);
			classify(new NaiveBayes(), entry.getValue(), labeledTest);
		}
	}

	public void classifyKNN(Map<String, Instances> trainDatasets, Instances unlabeledTest, Map<String, List<Double>> testLabels){
		// Read in test class values (labels/topics)
		List<String> classValues = Utility.loadLabelFile("test_noclass_rest.arff");
		for (Map.Entry<String, Instances> entry : trainDatasets.entrySet()) {
			String labelName = entry.getKey();
			Instances labeledTest = Utility.labelDataset(testLabels.get(labelName), unlabeledTest, labelName);
			System.out.println(labeledTest);
			classify(new IBk(5), entry.getValue(), labeledTest);
		}
	}

	private List<Double> classify(Classifier cls, Instances train, Instances test){
		List<Double> labels = new ArrayList<Double>();
		try {
			cls.buildClassifier(train);

			// evaluate classifier and print some statistics
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(cls, test);
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
			System.out.println(eval.toMatrixString());

			//			// label instances
			//			for (int i = 0; i < test.numInstances(); i++) {
			//				labels.add(cls.classifyInstance(test.instance(i)));
			//				//test.instance(i).setClassValue(clsLabel);
			//			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return labels;
	}

}
