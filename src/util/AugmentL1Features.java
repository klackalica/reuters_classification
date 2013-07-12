package util;

import java.util.Map;

import weka.core.Instances;

public class AugmentL1Features implements AugmentInput{

	private Map<String, Instances> l1features;

	public AugmentL1Features(Map<String, Instances> l1Features){
		this.l1features = l1Features;
		removeClassAttribute();
	}
	
	private void removeClassAttribute(){
		for(Map.Entry<String, Instances> e : l1features.entrySet()){
			Instances data = e.getValue();
			data.setClassIndex(-1);
			data.deleteAttributeAt(data.numAttributes()-1);
			l1features.put(e.getKey(), data);
		}
	}

	@Override
	public Map<String, Instances> augment(Map<String, Instances> datasets) {
		for(Map.Entry<String, Instances> e : datasets.entrySet()){
			Instances combined = l1features.get(e.getKey());
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

}
