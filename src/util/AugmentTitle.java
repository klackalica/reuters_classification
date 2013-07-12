package util;

import java.util.Map;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class AugmentTitle implements AugmentInput{

	private String filepath;
	private DatasetHelper dh;
	private StringToWordVector filter;

	public AugmentTitle(String filepath, DatasetHelper dh, StringToWordVector filter){
		this.filepath = filepath;
		this.dh = dh;
		this.filter = filter;
	}

	@Override
	public Map<String, Instances> augment(Map<String, Instances> datasets) {
		try{
			Instances data = dh.loadData(filepath);
			data.deleteAttributeAt(1);		// remove body attribute
			Instances titleData = Filter.useFilter(data, filter);

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

//	@Override
//	public Instances augmentTest(Map<String, Instances> test) {
//		
//	}

}
