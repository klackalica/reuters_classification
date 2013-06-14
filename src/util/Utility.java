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
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Utility {

	/**
	 * Load a dataset from a specified file.
	 * 
	 * @param filepath - path to a file
	 * @return Instances of a dataset
	 * @throws IOException
	 */
	public static Instances loadData(String filepath) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(filepath));
		Instances data = new Instances(reader);
		reader.close();
		return data;
	}

	/**
	 * Load all training datasets and convert them into a word vector format.
	 * 
	 * @param listOfFiles - Files from which to load datasets
	 * @param filter - StringToWordVector to use to convert loaded datasets into their word vector representations
	 * @return Map with (label name, dataset) entries
	 * @throws IOException
	 * @throws Exception
	 */
	public static Map<String, Instances> loadAllTrainDatasets(File[] listOfFiles, StringToWordVector filter) throws IOException, Exception{
		Map<String, Instances> trainDatasets = new HashMap<String, Instances>();
		for (File file : listOfFiles) {
			if (file.isFile() && !file.getName().endsWith("_rest.arff")) {
				System.out.println("Loading " + file.getAbsolutePath());
				// Convert to word vector.
				Instances unlabeledData = Filter.useFilter(loadData(file.getAbsolutePath()), filter);
				// Put (topic name, unlabeledData) into the map.
				trainDatasets.put(file.getName().split("\\.")[0], unlabeledData);
				//System.out.println(unlabeledData);
			}
		}
		return trainDatasets;
	}

	/**
	 * Load all labels from a specified file.
	 * 
	 * @param filepath - path to a file
	 * @return List of labels
	 */
	public static List<String> loadLabelFile(String filepath){
		BufferedReader br = null;
		List<String> labelValues = new ArrayList<String>();
		try {
			br = new BufferedReader(new FileReader(filepath));
			String strLine;
			while((strLine = br.readLine())!= null)
			{
				labelValues.add(strLine.trim());
			}
			br.close();
		} catch (FileNotFoundException e) {
			System.out.println("[Utility.loadLabelFile]: " + e.getMessage());
			e.printStackTrace();
		}
		catch (IOException e) {
			System.out.println("[Utility.loadLabelFile]: " + e.getMessage());
			e.printStackTrace();
		} 
		return labelValues;
	}

	/**
	 * Convert true test labels which are loaded from a file into a map representation.
	 * Map entries are (label name, list of numbers) pairs, where |list of numbers| = #test instances 
	 * and each list element either has value 0 or 1. 1 indicates that a test instance is labelled with
	 * the label name, 0 indicates otherwise.
	 * 
	 * @param labelsUsed - List of labels being considered in the classification (i.e. 20 most common ones)
	 * @param testLabels - Test labels in the same format they appear in the file
	 * @return Map of label names to a binary list of numbers
	 */
	public static Map<String, List<Double>> formatTestLabels(List<String> labelsUsed, List<String> testLabels){
		Map<String, List<Double>> realTestLabels = new HashMap<String, List<Double>>();

		for(String labelName : labelsUsed){
			List<Double> binaryLabels = new ArrayList<Double>();
			for(String instanceLabels : testLabels){
				if(instanceLabels.contains("not_"+labelName)){
					binaryLabels.add(0.0);
				}
				else{
					binaryLabels.add(1.0);
				}
			}
			realTestLabels.put(labelName, binaryLabels);
		}
		return realTestLabels;
	}

	/**
	 * Add a label column to the dataset and fill in label values for each instance.
	 * This method modifies the dataset passed to it.
	 * 
	 * @param filepath - File from which to load label values
	 * @param data - Dataset to add the label column
	 * @param labelName - name of the label to add (e.g. earn, acq, ship, wheat...)
	 */
	public static void labelDataset(String filepath, Instances data, String labelName){
		List<String> labelValues = loadLabelFile(filepath);
		
		// Add label attribute/column at the end of the dataset.
		List<String> possibleValues = new ArrayList<String>(
				Arrays.asList("1.0", "0.0"));
		data.insertAttributeAt(new Attribute("label_" + labelName, possibleValues), data.numAttributes());

		// Set label/class index
		data.setClassIndex(data.numAttributes()-1);

		// Add label values to the dataset.
		for (int i = 0; i < data.numInstances(); i++) {
			data.instance(i).setClassValue(Double.parseDouble(labelValues.get(i)));
		}
	}

	/**
	 * Add a label column to the dataset and fill in label values for each instance.
	 * This method creates a copy of the passed dataset and adds labels to it.
	 * Passed dataset remains unchanged.
	 * 
	 * @param labelValues - list of label values {0, 1} for each instance
	 * @param unlabeled  - Dataset to add the label column
	 * @param labelName - name of the label to add (e.g. earn, acq, ship, wheat...)
	 * @return Labeled dataset
	 */
	public static Instances labelDataset(List<Double> labelValues, Instances unlabeled, String labelName){
		Instances labeled = new Instances(unlabeled);

		// Add label attribute/column at the end of the dataset.
		List<String> possibleValues = new ArrayList<String>(
				Arrays.asList("1.0", "0.0"));
		labeled.insertAttributeAt(new Attribute("label_" + labelName, possibleValues), labeled.numAttributes());

		// Set label/class index
		labeled.setClassIndex(labeled.numAttributes()-1);

		// Add label values to the dataset.
		for (int i = 0; i < labeled.numInstances(); i++) {
			labeled.instance(i).setClassValue(labelValues.get(i));
		}

		return labeled;
	}
}
