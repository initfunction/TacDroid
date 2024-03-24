package PermDroid.staticguimodel;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import soot.jimple.infoflow.android.axml.AXmlNode;
import soot.jimple.infoflow.android.manifest.ProcessManifest;
import PermDroid.staticanalysis.utils.MappingConstants;

/**
 * ʹ���µĹ��췽ʽ��һ�ι�������ʹ��
 * private static IdProvider instance = null;
	private IdProvider() {}
	public static IdProvider v() {
		if(instance == null) {
			synchronized (IdProvider.class) {
				if(instance == null) {
					instance = new IdProvider();
				}
			}
		}
		return instance;
	}
 * 
 * */

public class AppInfo {
	public String packName;
	public List<String> activities=new ArrayList<String>();
	public List<String> danPermissoins=new ArrayList<String>();
	public ProcessManifest processManifest;
	public Set<String> needAPI=new HashSet<String>();    //�����������������  �����ϡ��ˣ������ڱ仯,���򻯰桱
    public String  mainActivityName;
	public Set<StepOneOutput>  apiLocateInfo =new HashSet<StepOneOutput>();   ///����ǵ�һ�� �õ���API��Ϣ
    public Set<StepOneOutput>  apiInfo22 =new HashSet<StepOneOutput>();    //���ǹ���STGʱ���õ�����Ϣ��  permission��û�п���
    /**
     * ���Խ�api��λ��Ϣ�浽����Ժ�ʹ�ã�����һ���ṹ��
     * 
     *  
     * */
    
    
	private static AppInfo instance = null;

	public static AppInfo v() {
		if(instance == null) {
			synchronized (AppInfo.class) {
				if(instance == null) {
					instance = new AppInfo();
				}
			}
		}
		return instance;
	}
    public AppInfo() {
    }
    
    
    public String getPackName() {
        return packName;
    }

    public void setPackName(String packName) {
        this.packName = packName;
    }

    public List<String> getActivities() {
        return activities;
    }

    public void setActivities(List<String> activities) {
        this.activities = activities;
    }
    
    public List<String> getDanPermissoins() {
		return danPermissoins;
	}

	public void setDanPermissoins(List<String>  danPermissoins) {
		this.danPermissoins = danPermissoins;
	}
	public ProcessManifest getProcessManifest() {
		return processManifest;
	}

	public void setProcessManifest(ProcessManifest processManifest) {
		this.processManifest = processManifest;
	} 
	
	
	
	public String getMainActivityName() {
		return mainActivityName;
	}
	public void setMainActivityName(String mainActivityName) {
		this.mainActivityName = mainActivityName;
	}
	//  get pkgname ,Activity name and danPermissions
    public void  getAppInfo(String apkPath) {
    	AppInfo appInfo = new AppInfo();
        try {
            processManifest = new ProcessManifest(apkPath);
            AXmlNode manifest = processManifest.getManifest();   // manifest node 
            Set<String> allPermissions=processManifest.getPermissions();     //get allpermissions  
            
            Iterator<String> itPermission =allPermissions.iterator();
            while(itPermission.hasNext()) {                               //-> get danpermissions
            	String P=itPermission.next();
            	for(int i =0;i<MappingConstants.needPermissions.length;i++) {
            		String needP=MappingConstants.needPermissions[i];
            		if (P.contains(needP)) {
            			instance.danPermissoins.add(needP);
            		}
            	}
            };
            String pack = manifest.getAttribute("package").getValue().toString();
            instance.packName = pack;
            List<AXmlNode> ACTs = processManifest.getActivities(); //Returns a list containing all nodes with tag activity.
            for (AXmlNode act : ACTs) {
                String actName = act.getAttribute("name").getValue().toString();
                instance.activities.add(actName);
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
        getneedMethodsignature();
        mainActivity();
        System.out.println(toString());
    }
    
    public void getneedMethodsignature() {
   	 //
    	MappingConstants.initMap1();
    	MappingConstants.initMap2();
    	if (instance.danPermissoins==null){
    		System.out.println("The app has no dangerous permissions,so it don't need to be tested");
    		return;
    		}
    	Iterator DP=instance.danPermissoins.iterator();
    	while(DP.hasNext()) {
    		String Pkey=(String) DP.next();
    		String[] API=MappingConstants.permissionToMethodSignatures.get(Pkey);
    		for (int i = 0; i < API.length; i++) {
    			instance.needAPI.add(API[i]);
			}
    	}
    	//System.out.println("getneedMethodsignature:  Pkg:"+instance.packName+"\tdanPer size:"+instance.danPermissoins.size()+"\tneedAPI size"+instance.needAPI.size());
    }
	@Override
    public String toString() {
		String string="PkgName:"+packName+"\n Dangerous permissions ";
		Iterator<String> it=danPermissoins.iterator();
		while (it.hasNext()) {
			string+="\t~"+it.next();
			
		}
		string=string+"   needAPI size:"+needAPI.size();
		string=string+" \n  MainActivity:"+mainActivityName;
        return string;
    }
	public void mainActivity() {
		List<AXmlNode> activity= processManifest.getActivities(); 
		//��ÿ����������ӱ�ǩ  ��intent-filter��-������ӱ�ǩ-���������action��ǩ-��������name ��ֵ��android.intent.action.MAIN
		for (AXmlNode act : activity) {
			String NAME = act.getAttribute("name").getValue().toString();
			List<AXmlNode>  intentFilter=act.getChildrenWithTag("intent-filter");
            if (intentFilter!=null ) {
            	for (AXmlNode intent:intentFilter) {  //
					List<AXmlNode> actioNodes=intent.getChildrenWithTag("action");
					for (AXmlNode action:actioNodes) {
						String actName=action.getAttribute("name").getValue().toString();
						if (actName.equals("android.intent.action.MAIN")) {
							mainActivityName=NAME;
							return;
						}
					}
				}
				
			}else {
				continue;
			}
        }
	}
}
