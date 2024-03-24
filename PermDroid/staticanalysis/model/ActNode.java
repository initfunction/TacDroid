package PermDroid.staticanalysis.model;

import PermDroid.staticanalysis.Manager;
import PermDroid.staticguimodel.NodeType;

public class ActNode extends FragNode {

	public ActNode(long id, String name) {
		super(id);
		this.setNodeType(NodeType.ACTIVITY);
		this.setName(name);
		Manager.v().putActNode(name, this);
	}

	private BaseNode leftDrawer;//The drawer slides out from the left
	private BaseNode rightDrawer;//The drawer slides out from the right

	public BaseNode getLeftDrawer() {
		return leftDrawer;
	}
	public void setLeftDrawer(BaseNode leftDrawer) {
		this.leftDrawer = leftDrawer;
	}
	public BaseNode getRightDrawer() {
		return rightDrawer;
	}
	public void setRightDrawer(BaseNode rightDrawer) {
		this.rightDrawer = rightDrawer;
	}
}
