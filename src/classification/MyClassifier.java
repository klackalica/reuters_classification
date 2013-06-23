package classification;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class MyClassifier {
	
	private String clsMethod;
	
	public MyClassifier(String clsMethod){
		this.clsMethod = clsMethod;
	}

//	public List<Double> classify(Instances train, Instances test){
//		if(clsMethod.equals("DT")){
//			return classify(new J48(), train, test);
//		}
//		else if(clsMethod.equals("NB")){
//			return classify(new NaiveBayes(), train, test);
//		}
//		else{
//			return classify(new IBk(5), train, test);
//		}
//	}
	
	public List<Double> classify(Classifier cls, Instances test){
		try {
			//cls.buildClassifier(train);

			// Predict labels
			List<Double> labels = new ArrayList<Double>();
			for (int i = 0; i < test.numInstances(); i++) {
				double predicted = cls.classifyInstance(test.instance(i));
				labels.add(predicted);
			}
			return labels;
		} catch (Exception e) {
			System.err.println("[MyClassifier.classify] Error: " + e.getMessage());
		}
		return null;
	}
}
