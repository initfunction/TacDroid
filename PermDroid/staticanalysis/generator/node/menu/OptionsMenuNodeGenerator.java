package PermDroid.staticanalysis.generator.node.menu;

import soot.SootMethod;
import PermDroid.staticanalysis.model.BaseNode;
import PermDroid.staticanalysis.utils.Logger;
import PermDroid.staticguimodel.NodeType;

public class OptionsMenuNodeGenerator extends AMenuNodeGenerator {
	
	public static BaseNode build(SootMethod sm) {
		Logger.i(TAG, "Start to generate OptionsMenuNode ["+sm.getSignature()+"]");
		return new OptionsMenuNodeGenerator(sm).generate();
	}
	
	private OptionsMenuNodeGenerator(SootMethod sm) {
		super(sm);
	}

	@Override
	protected BaseNode generate() {
		BaseNode optionsMenuNode = super.generate();
		optionsMenuNode.setNodeType(NodeType.OPTIONSMENU);
		return optionsMenuNode;
	}
}
