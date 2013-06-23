package util;

import java.util.HashMap;
import java.util.Map;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.Ranker;
import weka.core.Instances;

public class FeatureSelection {

	private int[] selectedAttributes;
	//private Instances originalTrainSet;
	private ASEvaluation eval;
	private ASSearch search;
	private int numToSelect;
	private weka.attributeSelection.AttributeSelection attsel;

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

	public int[] getSelectedFeatures(){
		return selectedAttributes;
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

	/**
	 * Performs feature selection on the original training dataset
	 * and filters the training datasets to have these features only.
	 * 
	 * @param trainDatasets - training datasets
	 * @return training datasets that contained only selected features
	 */
	public Map<String, Instances> selectFeatures(Map<String, Instances> trainDatasets, Instances originalTrain){
		// Do feature selection on the original training dataset
		attsel = new weka.attributeSelection.AttributeSelection();  // package weka.attributeSelection!
		attsel.setEvaluator(eval);
		attsel.setSearch(search);
		System.out.println("Selecting attributes...");
		try {
			attsel.SelectAttributes(originalTrain);
			String fsResults = attsel.toResultsString();
			Utility.outputToFile(fsResults);
			System.out.println(fsResults);

			selectedAttributes = attsel.selectedAttributes();		// obtain the attribute indices that were selected

			// Keep only selected features in the training datasets.
			Map<String, Instances> fstrainDatasets = new HashMap<String, Instances>();
			for(Map.Entry<String, Instances> e : trainDatasets.entrySet()){
				fstrainDatasets.put(e.getKey(), filterOutAttributes(e.getValue()));
			}
			return fstrainDatasets;
		} catch (Exception e) {
			System.err.println("[FeatureSelection.selectFeatures]: Error" + e.getMessage());
			return null;
		}
	}

	public Instances filterOutAttributes(Instances data){
		try {
			return attsel.reduceDimensionality(data);
		} catch (Exception e1) {
			System.err.println("[FeatureSelection.filterOutAttributes] Error: " + e1.getMessage());
			return null;
		}
	}
}
