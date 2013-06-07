package util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class DatasetIO {

	public Instances loadData() throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader("wheat_set.arff"));
		Instances data = new Instances(reader);
		reader.close();
		return str2wordVector(data);
	}
	
	private Instances str2wordVector(Instances data){
		StringToWordVector filter = new StringToWordVector();
		try {
			filter.setOptions(new String[]{"-R first-last", "-W 30000", "-prune-rate -1.0", "-I", "-N 0"});
			filter.setInputFormat(data);
			Instances dataset = Filter.useFilter(data, filter);
			// setting class attribute
			dataset.setClassIndex(0);
			return dataset;
		} catch (Exception e) {
			System.out.println("[DatasetIO.str2wordVector]: " + e.getMessage());
			//e.printStackTrace();
			return null;
		}
	}
}
