package classification;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import util.Utility;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class MyClassifier {
	
	private int seed = 1;
	private int folds = 10;
	private String clsMethod;
	private Map<String, Classifier> classifiers = new HashMap<String, Classifier>(); 
	
	public MyClassifier(String clsMethod){
		this.clsMethod = clsMethod;
	}
	
	public void trainClassifiers(Map<String, Instances> train){
		for (Map.Entry<String, Instances> e : train.entrySet()) {
			System.out.println("Training classifier for " + e.getKey() + "...");
			Classifier cls = trainClassifier(e.getValue());
			classifiers.put(e.getKey(), cls);
		}
	}

	private Classifier trainClassifier(Instances data){
		for(int i = 0; i < data.numInstances(); i++){
			double d = data.instance(i).classValue();
			assert d == 0.0 || d == 1.0 : "Class value is not 0 or 1!";
		}
		
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
			System.err.println("[MyClassifier.trainClassifier] Error: " + e.getMessage());
		}
		return cls;
	}
	
	public void crossValidateAll(Map<String, Instances> train){
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
			sb.append(evalAll.toMatrixString("Confusion Matrix"));
			sb.append("\n");

			Utility.outputDual(sb.toString());
		} catch (Exception e) {
			System.err.println("[MyClassifier.crossValidateData] Error: " + e.getMessage());
		}
	}
	
	public List<Double> classify(String labelName, Instances test){
		try {
			Classifier cls = classifiers.get(labelName);
			
			// Predict labels
			List<Double> labels = new ArrayList<Double>();
			for (int i = 0; i < test.numInstances(); i++) {
				double predicted = cls.classifyInstance(test.instance(i));
				assert predicted == 0.0 || predicted == 1.0 : "Predicted class value is not 0 or 1";
				labels.add(predicted);
			}
			return labels;
		} catch (Exception e) {
			System.err.println("[MyClassifier.classify] Error: " + e.getMessage());
		}
		return null;
	}
}
