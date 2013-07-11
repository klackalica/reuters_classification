package util;

import java.util.Map;

import weka.core.Instances;

public interface AugmentInput {

	public Map<String, Instances> augment(Map<String, Instances> train);
	//public Instances augmentTest(Map<String, Instances> train);
}
