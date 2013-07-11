package util;

import java.util.Map;

import weka.core.Instances;

public class AugmentL1Features implements AugmentInput{

	private Map<String, Instances> l1features;

	public AugmentL1Features(Map<String, Instances> l1Features){
		this.l1features = l1Features;
	}

	@Override
	public Map<String, Instances> augment(Map<String, Instances> datasets) {
//		// First remove class attribute, only want to keep features used in layer 1.
//		for(Map.Entry<String, Instances> e : datasets.entrySet()){
//			Instances data = e.getValue();
//			data.deleteAttributeAt(data.numAttributes()-1);
//			datasets.put(e.getKey(), data);
//		}

		for(Map.Entry<String, Instances> e : datasets.entrySet()){
			Instances fromL1 = l1features.get(e.getKey());
			Instances t = e.getValue();

			// Overwrite class attribute from layer1 features
			fromL1.insertAttributeAt(t.attribute(0), fromL1.numAttributes()-1);
			for(int i = 1; i < t.numAttributes(); i++){
				fromL1.insertAttributeAt(t.attribute(i), fromL1.numAttributes());

				// Add attribute values to the dataset.
				for (int j = 0; j < fromL1.numInstances(); j++) {
					double d = t.instance(j).value(i);
					fromL1.instance(j).setValue(fromL1.numAttributes()-1, d);
				}
			}
			// Set label/class index
			fromL1.setClassIndex(fromL1.numAttributes()-1);
			datasets.put(e.getKey(), fromL1);
			System.out.println("After adding l1 features: " + fromL1.numAttributes());
		}
		return datasets;
	}

}
