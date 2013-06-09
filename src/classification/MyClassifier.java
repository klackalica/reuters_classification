package classification;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import util.Utility;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;

//import weka.classifiers.lazy.IBk;
//import weka.classifiers.trees.J48;
//import weka.classifiers.bayes.NaiveBayes;

public class MyClassifier {

	public void classifyDecisionTree(Map<String, Instances> trainDatasets, Instances unlabeledTest){
		// Read in test class values (labels/topics)
		List<String> classValues = Utility.loadClassValues("test_noclass_rest.arff");
		for (Map.Entry<String, Instances> entry : trainDatasets.entrySet()) {
			Instances labeledTest = Utility.addClassToTestDataset(classValues, unlabeledTest, entry.getKey());
			System.out.println(labeledTest);
			classify(new J48(), entry.getValue(), labeledTest);
		}
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

	private void classify(Classifier cls, Instances train, Instances labeledTest){
		try {
			cls.buildClassifier(train);
			// evaluate classifier and print some statistics
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(cls, labeledTest);
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
			//System.out.println("Error rate: " + eval.errorRate());
			//System.out.println("PCt correct: " + eval.pctCorrect());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
