package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Utility {

	public static Instances loadData(String path) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(path));
		Instances data = new Instances(reader);
		reader.close();
		return data;
	}

	public static Map<String, Instances> loadAllTrainDatasets(File[] listOfFiles, StringToWordVector filter) throws IOException, Exception{
		Map<String, Instances> trainDatasets = new HashMap<String, Instances>();
		for (File file : listOfFiles) {
			if (file.isFile() && !file.getName().endsWith("_rest.arff")) {
				String abs = file.getAbsolutePath();
				System.out.println("Loading " + abs);
				Instances data = Filter.useFilter(loadData(file.getAbsolutePath()), filter);

				trainDatasets.put(file.getName().split("\\.")[0], data);
				//System.out.println(data);
			}
		}
		return trainDatasets;
	}

	public static Map<String, Instances> addClassToAllTrainDatasets(File[] listOfFiles, Map<String, Instances> trainDatasets){
		for (File file : listOfFiles) {
			if (file.isFile() && file.getName().endsWith("_rest.arff")) {
				String dataset_name =  file.getName().split("_")[0];
				addClassToTrainDataset(file.getAbsolutePath(), trainDatasets.get(dataset_name), dataset_name);
			}
		}
		return null;
	}

	public static List<String> loadClassValues(String filepath){
		BufferedReader br = null;
		List<String> classValues = new ArrayList<String>();
		try {
			br = new BufferedReader(new FileReader(filepath));
			String strLine;
			while((strLine = br.readLine())!= null)
			{
				classValues.add(strLine.trim());
			}
			br.close();
		} catch (FileNotFoundException e) {
			System.out.println("[Utility.addClassToTrainDataset]: " + e.getMessage());
			e.printStackTrace();
		}
		catch (IOException e) {
			System.out.println("[Utility.addClassToTrainDataset]: " + e.getMessage());
			e.printStackTrace();
		} 
		return classValues;
	}

	public static void addClassToTrainDataset(String filepath, Instances data, String classname){
		// Read in all class values
		List<String> classValues = loadClassValues(filepath);

		// Add class attribute to the dataset
		List<String> possibleValues = new ArrayList<String>(
				Arrays.asList("1", "0"));
		data.insertAttributeAt(new Attribute("class_" + classname, possibleValues), data.numAttributes());

		// Add class values to the dataset
		for (int i = 0; i < data.numInstances(); i++) {
			data.instance(i).setValue(data.numAttributes()-1, classValues.get(i));
		}
		// Set class index
		data.setClassIndex(data.numAttributes()-1);
	}

	public static Instances addClassToTestDataset(List<String> classValues, Instances unlabeled, String classname){
		// create copy
		Instances labeled = new Instances(unlabeled);

		// Add class attribute to the dataset
		List<String> possibleValues = new ArrayList<String>(
				Arrays.asList("1", "0"));
		labeled.insertAttributeAt(new Attribute("class_" + classname, possibleValues), labeled.numAttributes());

		// Add class values to the dataset
		for (int i = 0; i < labeled.numInstances(); i++) {
			String instanceLabels = classValues.get(i);
			String value = instanceLabels.contains("not_"+classname)? "0" : "1";
			labeled.instance(i).setValue(labeled.numAttributes()-1, value);
		}
		// Set class index
		labeled.setClassIndex(labeled.numAttributes()-1);
		return labeled;
	}
	
	public static Instances addClassToTestDataset(Instances unlabeled, String classname){
		// create copy
		Instances labeled = new Instances(unlabeled);

		// Add class attribute to the dataset
		List<String> possibleValues = new ArrayList<String>(
				Arrays.asList("1", "0"));
		labeled.insertAttributeAt(new Attribute("class_" + classname, possibleValues), labeled.numAttributes());

		// Set class index
		labeled.setClassIndex(labeled.numAttributes()-1);
		return labeled;
	}
}
