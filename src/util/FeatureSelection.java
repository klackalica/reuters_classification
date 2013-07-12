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

	private ASEvaluation eval;
	private ASSearch search;
	private int numToSelect;
	private Map<String, weka.attributeSelection.AttributeSelection> attselMap = new HashMap<String, weka.attributeSelection.AttributeSelection>();

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

	/**
	 * Performs feature selection on each of the training datasets
	 * and filters them to have these features only.
	 * 
	 * @param trainDatasets - training datasets
	 * @return training datasets that contained only selected features
	 */
	public Map<String, Instances> selectFeatures(Map<String, Instances> trainDatasets){
		Map<String, Instances> fstrainDatasets = new HashMap<String, Instances>();
		for(Map.Entry<String, Instances> e : trainDatasets.entrySet()){
			// Keep only selected features in the training datasets.
			fstrainDatasets.put(e.getKey(), selectFeaturesOnDataset(e.getValue(), e.getKey()));
		}
		return fstrainDatasets;
	}

	public Instances selectFeaturesOnDataset(Instances dataset, String name){
		weka.attributeSelection.AttributeSelection attsel = new weka.attributeSelection.AttributeSelection();  // package weka.attributeSelection!

		attsel.setEvaluator(eval);
		attsel.setSearch(search);
		Utility.outputDual("Selecting attributes for " + name + "...");
		try {
			attsel.SelectAttributes(dataset);
			String fsResults = attsel.toResultsString();
			Utility.outputDual(fsResults);

			attselMap.put(name, attsel);

			// Keep only selected features in the training datasets.
			return filterOutAttributes(dataset, name);
		} catch (Exception e1) {
			System.err.println("[FeatureSelection.selectFeaturesOnDataset]: Error" + e1.getMessage());
			return null;
		}
	}

	public Instances filterOutAttributes(Instances data, String labelName){
		try {
			return attselMap.get(labelName).reduceDimensionality(data);
		} catch (Exception e1) {
			System.err.println("[FeatureSelection.filterOutAttributes] Error: " + e1.getMessage());
			return null;
		}
	}
}
