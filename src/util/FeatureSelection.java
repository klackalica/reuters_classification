package util;

import java.io.IOException;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class FeatureSelection {

	private int[] selectedAttributes;
	private Instances originalTrainSet;
	private ASEvaluation eval;
	private ASSearch search;
	private int numToSelect;
	
	public FeatureSelection(int fsMethod, int num){
		numToSelect = num;
		if(fsMethod == 1){
			CfsSubsetGreedyStepwise();
		}
		else if(fsMethod == 2){
			CfsSubsetBestFirst();
		}
		else if(fsMethod == 3){
			GainRatioRanker();
		}
	}
	
	public void CfsSubsetGreedyStepwise(){
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		this.search = search;
		this.eval = eval;
	}
	
	public void CfsSubsetBestFirst(){
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();
		this.search = search;
		this.eval = eval;
	}

	public void GainRatioRanker(){
		GainRatioAttributeEval eval = new GainRatioAttributeEval();
		Ranker search = new Ranker();
		search.setNumToSelect(numToSelect);
		this.search = search;
		this.eval = eval;
	}

	public void selectAttributes(){
		fsOnOriginalTrain();
		
		weka.attributeSelection.AttributeSelection attsel = new weka.attributeSelection.AttributeSelection();  // package weka.attributeSelection!
		attsel.setEvaluator(eval);
		attsel.setSearch(search);
		System.out.println("Selecting attributes...");
		try {
			attsel.SelectAttributes(originalTrainSet);
			String fsResults = attsel.toResultsString();
			Utility.outputToFile(fsResults);
			System.out.println(fsResults);
			
			selectedAttributes = attsel.selectedAttributes();		// obtain the attribute indices that were selected
			Utility.outputToFile("\nNumber attributes selected: " + attsel.numberAttributesSelected());
			System.out.println("\nNumber attributes selected: " + attsel.numberAttributesSelected());
		} catch (Exception e) {
			System.out.println("[FeatureSelection.selectAttributes]: " + e.getMessage());
		}
	}
	
	public Instances filterOutAttributes(Instances dataset){
		// Keep the class attribute
		int[] copy = new int[selectedAttributes.length + 1];
		System.arraycopy( selectedAttributes, 0, copy, 0, selectedAttributes.length );
		copy[copy.length - 1] = dataset.classIndex();
		System.out.println(dataset);
		Remove remove = new Remove();						// new instance of filter
		try {
			remove.setAttributeIndicesArray(copy);
			remove.setInvertSelection(true);
			remove.setInputFormat(dataset);					// inform filter about dataset **AFTER** setting options
			Instances fltr = Filter.useFilter(dataset, remove);		// apply filter
			// set class index!!
			//
			return fltr;
		} catch (Exception e) {
			System.out.println("[FeatureSelection.filterOutAttributes]: " + e.getMessage());
			return null;
		}
	}

	private void fsOnOriginalTrain(){
		Instances train;
		try {
			train = DatasetHelper.loadData("alltrain_class.arff");
			StringToWordVector filter = DatasetHelper.createWordVectorFilter(train);

			originalTrainSet = Filter.useFilter(train, filter);
			originalTrainSet.setClassIndex(0);
			//System.out.println("num attributes = " + originalTrainSet.numAttributes());
		} catch (IOException e) {
			System.out.println("[FeatureSelection.fsOnOriginalTrain]: " + e.getMessage());
		} catch (Exception e) {
			System.out.println("[FeatureSelection.fsOnOriginalTrain]: " + e.getMessage());
		}
		
	}
}
