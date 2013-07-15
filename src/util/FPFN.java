package util;

public class FPFN {

	private String labelName;
	private int FP = 0;
	private int FN = 0;
	
	public FPFN(String label){
		labelName = label;
	}
	
	public void incrementFP(){
		FP++;
	}
	
	public void incrementFN(){
		FN++;
	}
	
	public int getFP(){
		return FP;
	}
	
	public int getFN(){
		return FN;
	}
	
	@Override
	public String toString(){
		return labelName + ":\n\t" + FP + "\n\t" + FN;
	}
}
