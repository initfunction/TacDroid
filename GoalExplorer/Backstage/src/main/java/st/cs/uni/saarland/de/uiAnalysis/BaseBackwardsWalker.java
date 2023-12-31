package st.cs.uni.saarland.de.uiAnalysis;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import soot.Body;
import soot.SootMethod;
import st.cs.uni.saarland.de.searchTabs.TabInfo;
import st.cs.uni.saarland.de.dissolveSpecXMLTags.TabViewInfo;
import st.cs.uni.saarland.de.helpClasses.Helper;
import st.cs.uni.saarland.de.helpClasses.MyStmtSwitchForResultLists;
import st.cs.uni.saarland.de.helpMethods.IterateOverUnitsHelper;
import st.cs.uni.saarland.de.searchScreens.LayoutInfo;
import st.cs.uni.saarland.de.searchScreens.StmtSwitchForLayoutInflater;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Created by avdiienko on 11/05/16.
 */
public class BaseBackwardsWalker implements Runnable {
    protected Set<TabViewInfo> resultTabsViews = new HashSet<>();
    protected Set<TabInfo> resultTabs = new HashSet<>();
    protected Map<Integer, LayoutInfo> resultLayouts = new HashMap<>();
    private final Class<? extends MyStmtSwitchForResultLists> switchClass;
    private final Logger logger =  LoggerFactory.getLogger(Thread.currentThread().getName());

    public Set<TabViewInfo> getResultTabViewInfos() {
        return resultTabsViews;
    }

    public Set<TabInfo> getResultTabInfos() {
        return resultTabs;
    }

    public Map<Integer, LayoutInfo> getResultLayoutInfos() {
        return resultLayouts;
    }

    private final SootMethod currentMethod;
    private final String callerSootClass;

    public BaseBackwardsWalker(Class<? extends MyStmtSwitchForResultLists> requiredClass, SootMethod currentMethod){
        this(requiredClass, currentMethod, null);
    }

    public BaseBackwardsWalker(Class<? extends MyStmtSwitchForResultLists> requiredClass, SootMethod currentMethod, String callerSootClass) {
        switchClass = requiredClass;
        this.currentMethod = currentMethod;
        this.callerSootClass = callerSootClass;
    }

    @Override
    public void run() {
        if (!currentMethod.hasActiveBody()) {
            return;
        }
        final Body body = currentMethod.getActiveBody();
        if (!Helper.processMethod(body.getUnits().size()) || !Helper.isClassInAppNameSpace(body.getMethod().getDeclaringClass().getName())) {
            return;
        }
        try {

            final MyStmtSwitchForResultLists stmtSwitch = this.switchClass.getConstructor(SootMethod.class).newInstance((SootMethod) null);

            stmtSwitch.init();
            if(switchClass.equals(StmtSwitchForLayoutInflater.class)){
                stmtSwitch.setCallerSootClass(callerSootClass);
            }
            stmtSwitch.setCurrentSootMethod(body.getMethod());
            IterateOverUnitsHelper.newInstance().runUnitsOverMethodBackwards(body, stmtSwitch);

            Set<TabViewInfo> resTV = stmtSwitch.getResultedTabsViews();
            Set<TabInfo> resT = stmtSwitch.getResultedTabs();
            Map<Integer, LayoutInfo> resL = stmtSwitch.getResultLayoutInfos();

            if (resT != null) {
                String toPrint = currentMethod.getSignature() + " basebackwards tabs: ";
                for (TabInfo tab : stmtSwitch.getResultedTabs()) {
                    toPrint += " " + tab.toString();
                }
                //logger.info(toPrint);
                resultTabs.addAll(resT);
                toPrint = currentMethod.getSignature() + " basebackwards tabs after adding: ";
                for (TabInfo tab : getResultTabInfos()) {
                    toPrint += " " + tab.toString();
                }
                //logger.info(toPrint);
            }
            if (resTV != null)
                resultTabsViews.addAll(resTV);
            if (resL != null)
                resultLayouts.putAll(resL);


        } catch (Exception e) {
            e.printStackTrace();
            Helper.saveToStatisticalFile(Helper.exceptionStacktraceToString(e));
        }

    }
}
