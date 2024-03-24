package PermDroid.staticanalysis;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import mySQL.EdgeDB;
import mySQL.WidgetDB;
import mySQL.WindowDB;
import mySQL.clearDB;
import soot.Body;
import soot.SootClass;
import soot.SootMethod;
import soot.Unit;
import soot.jimple.Stmt;
import PermDroid.staticanalysis.analyzer.EventAnalyzer;
import PermDroid.staticanalysis.generator.node.ActivityNodeGenerator;
import PermDroid.staticanalysis.generator.node.FragmentNodeGenerator;
import PermDroid.staticanalysis.model.ActNode;
import PermDroid.staticanalysis.model.BaseNode;
import PermDroid.staticanalysis.model.FragNode;
import PermDroid.staticanalysis.model.Transition;
import PermDroid.staticanalysis.utils.ClassService;
import PermDroid.staticanalysis.utils.IOService;
import PermDroid.staticanalysis.utils.IdProvider;
import PermDroid.staticanalysis.utils.Logger;
import PermDroid.staticanalysis.utils.Utils;
import PermDroid.staticguimodel.Graph;
import PermDroid.staticguimodel.Node;
import PermDroid.staticguimodel.NodeType;
import PermDroid.staticguimodel.Widget;
import PermDroid.staticguimodel.Edge;


public class Main {
	private static final String TAG = "[Main]";
	public static String androidPlatformLocation = "/Users/luyanchen/Library/Android/sdk/platforms";
	public static String apkDir = "/Users/luyanchen/Desktop/PermDroid-master/Static/";//F:\\APK-ysh\\socialApps33\\
	public static String apkName = "download";//timberfoss_21
	public static String apk = apkDir+apkName+".apk";
	//public static File totalRes = new File("G:\\APK\\result\\totalRes.txt");  
	
	
	public static void main(String[] args) {
//		if (staticTepOne.v().getAPILocation()==true) {
//				runApp();
//		}
		runApp();
	}
	public static void runApp() {
		Main m = new Main();
		long startTime = System.currentTimeMillis();
		m.analysis();
		long endTime = System.currentTimeMillis();
        long executeTime = (endTime - startTime)/1000;
		Logger.i(TAG, "End, took "+executeTime+"s");
		Logger.i(TAG, "LOC: "+AppParser.v().loc());
	}
	public void analysis() {
		AppParser.v().init(apkDir, apkName);
		nodesGenerate();
		transitionsGenerate();
		buildGraph();
	}
	private void nodesGenerate() {
		
		Logger.i(TAG, "============ Fragment Nodes ============");
		for(SootClass frag : AppParser.v().getFragments()) {
			FragmentNodeGenerator.build(frag);
		}
		Logger.i(TAG, "============ Fragment Nodes ============");
		Logger.i(TAG, "============ Activity Nodes ============");
		for(SootClass act : AppParser.v().getActivities()) {
			ActivityNodeGenerator.build(act);
		}
		Logger.i(TAG, "============ Activity Nodes =============");
	}
	
	private void transitionsGenerate() {
		Logger.i(TAG, "============ Transitions ============");
		for(BaseNode baseNode : Manager.v().getNodes()) { 
			if(baseNode.getNodeType().equals(NodeType.DIALOG)) {
				List<Widget> ws = baseNode.getWidgets();
				for(Widget w : ws) {
					if(w.getEventHandler() != null) {
						EventAnalyzer.analysis(w.getEventHandler(), baseNode, w);
					}
				}
			}
		}
		
		for(FragNode fragNode : Manager.v().getFragNodes()) {
			if(fragNode.getOptionsMenu() != null) {
				BaseNode optionsMenu = fragNode.getOptionsMenu();
				Transition t = new Transition();
				t.setId(IdProvider.v().edgeId());
				t.setSrc(fragNode.getId());
				t.setTgt(optionsMenu.getId());
				t.setLabel("Open OptionsMenu");
				//ȱ��add�ɣ��Ҽӵ�
				Manager.v().add(t);
			}
			for(String fragName : fragNode.getFragmentsName()) {
				FragNode tgtFragNode = Manager.v().getFragNodeByName(fragName);
				if(tgtFragNode != null && Manager.v().hasTransition(fragNode.getId(), tgtFragNode.getId()))
					continue;
				Transition t = new Transition();
				t.setId(IdProvider.v().edgeId());
				t.setSrc(fragNode.getId());
				if(tgtFragNode != null) {
					t.setTgt(tgtFragNode.getId());
					t.setLabel("Load");
				}else {
					t.setLabel("MTN["+fragName+"]");
				}
				Manager.v().add(t);
			}
			
			List<Widget> ws = fragNode.getWidgets();
			for(Widget w : ws) {
				if(w.getEventHandler() != null) {
					EventAnalyzer.analysis(w.getEventHandler(), fragNode, w); 
				}else if(w.getEventMethod() != null && w.getEventMethod().equals(Utils.OPENCONTEXTMENU)) {//�����ô���ܰ���
					BaseNode contextMenu = fragNode.getContextMenu();
					if(contextMenu != null) {
						Transition t = new Transition();
						t.setId(IdProvider.v().edgeId());
						t.setTgt(contextMenu.getId());
						t.setSrc(fragNode.getId());
						t.setWidget(w);
						Manager.v().add(t);
					}
				}
			}
		}
		
		for(ActNode actNode : Manager.v().getActNodes()) {
			List<Widget> ws = actNode.getWidgets();
			for(Widget w : ws) {
				if(w.getEventHandler() != null) {
					EventAnalyzer.analysis(w.getEventHandler(), actNode, w);
				}else if(w.getEventMethod() != null && w.getEventMethod().equals(Utils.OPENCONTEXTMENU)) {
					BaseNode contextMenu = actNode.getContextMenu();
					if(contextMenu == null)
						continue;
					Transition t = new Transition();
					t.setId(IdProvider.v().edgeId());
					t.setTgt(contextMenu.getId());
					t.setSrc(actNode.getId());
					t.setWidget(w);
					//ȱ��add�ɣ��Ҽӵ�
					Manager.v().add(t);
				}
			}
			//���� Activity��OptionsMenu�ı�
			if(actNode.getOptionsMenu() != null) {
				BaseNode optionsMenu = actNode.getOptionsMenu();
				Transition t = new Transition();
				t.setId(IdProvider.v().edgeId());
				t.setSrc(actNode.getId());
				t.setTgt(optionsMenu.getId());
				t.setLabel("Open OptionsMenu");
				//ȱ��add�ɣ��Ҽӵ�
				Manager.v().add(t);
			}
			//���� Activity��Drawer�ı�
			if(actNode.getLeftDrawer() != null) {
				BaseNode leftDrawer = actNode.getLeftDrawer();
				Transition t = new Transition();
				t.setId(IdProvider.v().edgeId());
				t.setSrc(actNode.getId());
				t.setTgt(leftDrawer.getId());
				t.setLabel("Open left Drawer");
				//ȱ��add�ɣ��Ҽӵ�               
				Manager.v().add(t);
			}
			if(actNode.getRightDrawer() != null) {
				BaseNode rightDrawer = actNode.getLeftDrawer();
				Transition t = new Transition();
				t.setId(IdProvider.v().edgeId());
				t.setSrc(actNode.getId());
				t.setTgt(rightDrawer.getId());
				t.setLabel("Open right Drawer");
				//ȱ��add�ɣ��Ҽӵ�
				Manager.v().add(t);
			}
			//���� Activity�е�Fragment��OptinsMenu | Drawer�ı�  
			//�Լ� Activity���ܵ���Fragment�ı�
			if(!actNode.getFragmentsName().isEmpty()) {
				BaseNode optionsMenu = actNode.getOptionsMenu();
				BaseNode leftDrawer = actNode.getLeftDrawer();
				BaseNode rightDrawer = actNode.getLeftDrawer();
				for(String fragName : actNode.getFragmentsName()) {
					FragNode fragNode = Manager.v().getFragNodeByName(fragName);//������ҵ���
					if(fragNode == null || Manager.v().hasTransition(actNode.getId(), fragNode.getId()))
						continue;
					//Activity���ܵ���Fragment�ı�
					Transition t1 = new Transition();
					t1.setId(IdProvider.v().edgeId());
					t1.setSrc(actNode.getId());
					t1.setTgt(fragNode.getId());
					t1.setLabel("Load");
					//ȱ��add�ɣ��Ҽӵ�
					Manager.v().add(t1);
					//���� Activity�е�Fragment��OptinsMenu | Drawer�ı�
					if(optionsMenu != null || leftDrawer != null || rightDrawer != null) {
						if(optionsMenu != null) {
							Transition t = new Transition();
							t.setId(IdProvider.v().edgeId());
							t.setSrc(fragNode.getId());
							t.setTgt(optionsMenu.getId());
							t.setLabel("Open OptionsMenu");
							//ȱ��add�ɣ��Ҽӵ�
							Manager.v().add(t);
						}
						if(leftDrawer != null) {
							Transition t = new Transition();
							t.setId(IdProvider.v().edgeId());
							t.setSrc(fragNode.getId());
							t.setTgt(leftDrawer.getId());
							t.setLabel("Open left Drawer");
							//ȱ��add�ɣ��Ҽӵ�
							Manager.v().add(t);
						}
						if(rightDrawer != null) {
							Transition t = new Transition();
							t.setId(IdProvider.v().edgeId());
							t.setSrc(fragNode.getId());
							t.setTgt(rightDrawer.getId());
							t.setLabel("Open right Drawer");
							//ȱ��add�ɣ��Ҽӵ�
							Manager.v().add(t);
						}
					}
				}
			}
		}
		Logger.i(TAG, "=====================================");
	}
	
	private void buildGraph() {
		Graph graph = Manager.v().buildGraph();
		writeResult(graph);
	}
	
	private void writeResult(Graph graph) {
		IOService.v().writeResult(graph);  	// .bat
		IOService.v().writeResultCsv(graph);   // *.csv
		StringBuilder sb = new StringBuilder();
		sb.append("Node size: ").append(graph.getNodeSize());
		sb.append("widget size: ").append(graph.getWidgets().size());
		sb.append("Edge size: ").append(graph.getEdges().size()).append("\n");
		
		StringBuilder sb1 = new StringBuilder();
		sb1.append(sb).append(graph.toString());
		IOService.v().writeResultString(sb1.toString());  //gSTG.txt
		
		System.out.println("-----------------Result-----------------");
		System.out.println(sb.toString());
	}


	private void writer(File file, String text, boolean append) throws IOException {
		if(!file.exists() || !file.isFile()){
			file.createNewFile();
		}
		try(FileOutputStream fileOutputStream = new FileOutputStream(file, append);
            OutputStreamWriter outputStreamWriter = new OutputStreamWriter(fileOutputStream);
            BufferedWriter bufferedWriter = new BufferedWriter(outputStreamWriter)){
            bufferedWriter.append(text).append("\r\n");
		}
	}
	
	
	
	// *****useless code is put here.*****
	
//	public static void runApps() {
//	String mApkDir = "F:\\00AAZZG\\0TestAPK\\Fdroid\\4\\";
//	File dir = new File(mApkDir);
//	if(!dir.exists() || !dir.isDirectory()) {
//		Logger.i(TAG, dir.getAbsolutePath()+" is not a directory");
//		System.exit(0);
//	}
//	
//	File[] apkFiles = dir.listFiles(new FileFilter() {
//		
//		@Override
//		public boolean accept(File arg0) {
//			if(arg0.getAbsolutePath().endsWith(".apk"))
//				return true;
//			return false;
//		}
//	});
//	Main m = new Main();
//	for(File apkFile : apkFiles) {
//		String mApkName = apkFile.getName().replace(".apk", "");
//		System.out.println(apkFile.getName());
//		try {
//			m.analysis_(mApkDir, mApkName);
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
//	}
////	try {
////		m.writer(totalRes, String.valueOf(size), true);
////	} catch (IOException e) {
////		e.printStackTrace();
////	}
//}
//public void analysis_(String mApkDir, String mApkName) throws Exception {
//	writer(totalRes, "===============================", true);
//	writer(totalRes, "APK: "+mApkName, true);
//	Logger.i(TAG, "===============================");
//	Logger.i(TAG, "DIR: "+mApkDir);
//	Logger.i(TAG, "APK: "+mApkName);
//	Logger.i(TAG, "Start to build gSTG...");
//	AppParser.reset();
//	long startTime = System.currentTimeMillis();
//	AppParser.v().init(mApkDir, mApkName);
//	nodesGenerate();
//	transitionsGenerate();
//	buildGraph();
//	long endTime = System.currentTimeMillis();
//    long executeTime = (endTime - startTime)/1000;
//	Logger.i(TAG, "End, took "+executeTime+"s");
//	Logger.i(TAG, "LOC: "+AppParser.v().loc());
//	Logger.i(TAG, "===============================");
//	writer(totalRes, "End, took "+executeTime+"s", true);
//	writer(totalRes, "LOC: "+AppParser.v().loc()+"\n", true);
//	writer(totalRes, "===============================", true);
//}
}
