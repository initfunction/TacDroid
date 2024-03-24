package PermDroid.staticguimodel;

import java.util.Iterator;

import soot.SootClass;
import PermDroid.staticanalysis.AppParser;

public class StepOneOutput {
	public String CLASS;  
	public String METHOD;
	public String APISIGNATURE;
	public String PERMISSIONS;
	
	/**
	 * ___***start***____
		Class :de.uni_potsdam.hpi.openmensa.extension.LocationManagerKt
		Method :boolean  requestLocationUpdatesIfSupported(android.location.LocationManager,java.lang.String,long,float,android.location.LocationListener)
		API signature:<android.location.LocationManager: void requestLocationUpdates(java.lang.String,long,float,android.location.LocationListener)>
		permission :"android.permission.ACCESS_FINE_LOCATION  android.permission.ACCESS_COARSE_LOCATION  "
		____***end***_____
	 * */
	public String inAct() {
		

		Iterator<SootClass> iterator=AppParser.v().getActivities().iterator();
		while (iterator.hasNext()) {
			SootClass sootClass = (SootClass) iterator.next();
			if (CLASS.contains(sootClass.getName())) { 
				return sootClass.getName();
			}
		}
		
		return "NO";
		
		
	}
	
public String inFrag() {
		

		Iterator<SootClass> iterator1=AppParser.v().getFragments().iterator();
		while (iterator1.hasNext()) {
			SootClass sootClass = (SootClass) iterator1.next();
			if (CLASS.contains(sootClass.getName())) { 
				return sootClass.getName();
			}
		}
		return "NO";
		
		
	}
	
	public  boolean isEqual(StepOneOutput stg) {
		if (CLASS.equals(stg.CLASS) && METHOD.equals(stg.METHOD)
				&& APISIGNATURE.equals(stg.APISIGNATURE)) {
			return true;
			
		}
		return false; 
	}
	
	
	
}
