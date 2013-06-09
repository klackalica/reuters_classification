package util;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class FeatureSelection {

	public Instances CfsSubsetGreedyStepwise(Instances dataset){
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		return apply(eval, search, dataset);
	}
	
	public Instances GainRatioRanker(Instances dataset){
		GainRatioAttributeEval eval = new GainRatioAttributeEval();
		Ranker search = new Ranker();
		return apply(eval, search, dataset);
	}
	
	private Instances apply(ASEvaluation eval, ASSearch search, Instances dataset){
		AttributeSelection filter = new AttributeSelection();  // package weka.filters.supervised.attribute!
		
		filter.setEvaluator(eval);
		filter.setSearch(search);
		try {
			filter.setInputFormat(dataset);
			// generate new data
			System.out.println("[FeatureSelection.apply]\tSearching for best features...");
			Instances reduces_dataset = Filter.useFilter(dataset, filter);
			//System.out.println(reduces_dataset);
			return reduces_dataset;
		} catch (Exception e) {
			System.out.println("[FeatureSelection.apply]: " + e.getMessage());
			//e.printStackTrace();
			return null;
		}
	}
}
