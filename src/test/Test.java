package test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import util.DatasetHelper;
import util.FeatureSelection;
import util.Utility;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Instances;

public class Test {

	public static void main(String[] args) throws Exception{
		DatasetHelper dh = new DatasetHelper(5000);
		Instances data = dh.loadData("for_testing.arff");
		data.setClassIndex(4);
		FeatureSelection fs = new FeatureSelection(2, 500);

		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();
		// Do feature selection on the original training dataset
		weka.attributeSelection.AttributeSelection attsel = new weka.attributeSelection.AttributeSelection();  // package weka.attributeSelection!
		attsel.setEvaluator(eval);
		attsel.setSearch(search);

		System.out.println("Selecting attributes...");
		attsel.SelectAttributes(data);
		String fsResults = attsel.toResultsString();
		//System.out.println(fsResults);

		int[] selectedAttributes = attsel.selectedAttributes();		// obtain the attribute indices that were selected

		Instances fltr = attsel.reduceDimensionality(data);
		System.out.println("Data ----------------------------:\n" + data);
		System.out.println("Fltr ----------------------------:\n" + fltr);
		System.out.println("fltr class index is " + fltr.classIndex());
		
		List<Double> old = new ArrayList<Double>();
		List<Double> nw = new ArrayList<Double>();
		for(int i = 0; i < fltr.numInstances(); i++){
			old.add(data.instance(i).classValue());
			nw.add(fltr.instance(i).classValue());
		}
		System.out.println(old);
		System.out.println(nw);
		for(int i = 0; i < nw.size(); i++){
			System.out.println("Comparing " + old.get(i) + " with " + nw.get(i));
			System.out.println(old.get(i).doubleValue() == nw.get(i).doubleValue());
		}
		
		List<Integer> labelsUsed = new ArrayList<Integer>(
				Arrays.asList(0,0,1,1,0,0,0,0,0,0,1));
		System.out.println(labelsUsed.toString());
	}
}
