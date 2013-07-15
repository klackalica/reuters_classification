package util;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class AugmentFSTitle implements AugmentInput{

	private String dataPath;
	private String labelPath;
	private DatasetHelper dh;
	private StringToWordVector filter;
	private FeatureSelection fs;
	private List<String> possibleLabels;
	
	public AugmentFSTitle(String dataPath, String labelPath, DatasetHelper dh, StringToWordVector filter, List<String> possibleLabels){
		this.dataPath = dataPath;
		this.labelPath = labelPath;
		this.dh = dh;
		this.filter = filter;
		this.fs = new FeatureSelection(3, 2000);
		this.possibleLabels = possibleLabels;
	}
	
	@Override
	public Map<String, Instances> augment(Map<String, Instances> datasets) {
		try{
			Instances data = dh.loadData(dataPath);
			data.deleteAttributeAt(1);		// remove body attribute
			Instances titleData = Filter.useFilter(data, filter);
			
			// Labelling
			dh.labelDataset(labelPath, titleData, "title_data", possibleLabels);
			
			// Feature selection
			titleData = fs.selectFeaturesOnDataset(titleData, "title");
			System.out.println("title data num attr: " + titleData.numAttributes());
			System.out.println(titleData);
			// Remove class attribute
			titleData.setClassIndex(-1);
			titleData.deleteAttributeAt(titleData.numAttributes()-1);

			for(Map.Entry<String, Instances> e : datasets.entrySet()){
				Instances combined = new Instances(titleData);
				Instances toAdd = e.getValue();
				for(int i = 0; i < toAdd.numAttributes(); i++){
					combined.insertAttributeAt(toAdd.attribute(i), combined.numAttributes());

					// Add attribute values to the dataset.
					for (int j = 0; j < combined.numInstances(); j++) {
						double d = toAdd.instance(j).value(i);
						combined.instance(j).setValue(combined.numAttributes()-1, d);
					}
				}
				// Set label/class index
				combined.setClassIndex(combined.numAttributes()-1);
				datasets.put(e.getKey(), combined);
			}
			return datasets;
		}
		catch(Exception e){
			System.err.println("[AugmentTitle.augment]: " + e.getMessage());
			return null;
		}
	}

}
