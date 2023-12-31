package st.cs.uni.saarland.de.uiAnalysis;

import soot.SootMethod;
import st.cs.uni.saarland.de.helpMethods.CastInfoHelper;
import st.cs.uni.saarland.de.searchMenus.MenuInfo;
import st.cs.uni.saarland.de.searchMenus.StmtSwitchForOptionMenus;

import java.util.Set;
import java.util.Map;

/**
 * Created by avdiienko on 11/05/16.
 */
public class ContextMenusFinder implements Runnable {
    private Set<MenuInfo> contextMenus;
    private final SpecMethodsForwardWalker specMethodsForwardWalker;

    public ContextMenusFinder(SootMethod currentSootMethod){
        specMethodsForwardWalker = new SpecMethodsForwardWalker(StmtSwitchForOptionMenus.class, currentSootMethod, "onCreateContextMenu");
    }

    public ContextMenusFinder(SootMethod currentSootMethod, Map<String, String> dynStrings){
        specMethodsForwardWalker = new SpecMethodsForwardWalker(StmtSwitchForOptionMenus.class, currentSootMethod, "onCreateContextMenu", dynStrings);
    }

    @Override
    public void run() {
        specMethodsForwardWalker.run();
        
        contextMenus = CastInfoHelper.getInstance().getResultsInMenuInfo(specMethodsForwardWalker.getResults());
    }

    public Set<MenuInfo> getContextMenus(){
        return contextMenus;
    }
}
