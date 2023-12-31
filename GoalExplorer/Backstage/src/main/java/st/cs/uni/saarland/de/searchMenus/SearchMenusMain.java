package st.cs.uni.saarland.de.searchMenus;

import java.util.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import st.cs.uni.saarland.de.entities.Application;
import st.cs.uni.saarland.de.entities.AppsUIElement;
import st.cs.uni.saarland.de.entities.Listener;
import st.cs.uni.saarland.de.entities.Menu;
import st.cs.uni.saarland.de.entities.SpecialXMLTag;
import st.cs.uni.saarland.de.entities.XMLLayoutFile;
import st.cs.uni.saarland.de.helpClasses.Helper;
import st.cs.uni.saarland.de.helpMethods.CheckIfMethodsExisting;
import st.cs.uni.saarland.de.testApps.AppController;
import st.cs.uni.saarland.de.testApps.Content;
import st.cs.uni.saarland.de.searchDynDecStrings.DynDecStringInfo;

public class SearchMenusMain {
	/*private Set<DynDecStringInfo> dynStringInfos;

	public SearchMenusMain(Set<DynDecStringInfo> dynStringInfos){
		this.dynStringInfos = dynStringInfos;
	}*/

	private final Logger logger =  LoggerFactory.getLogger(Thread.currentThread().getName());
	private Content content = Content.getInstance();
	private AppController appController = AppController.getInstance();

	public void getMenusInApp(Set<MenuInfo> optionInfos, Set<MenuInfo> contextMenuInfos,
							  Set<MenuInfo> onCreateContextInfos, Set<PopupMenuInfo> popupInfos,
							  Set<DropDownNavMenuInfo> dropDownInfos){
		
		try{
			logger.debug("Getting option menus");
			getOptionMenus(optionInfos);
		}catch(Exception e){
			e.printStackTrace();
			logger.error("OptionMenus: " + e);
		}
		try{
			logger.debug("Getting contextual menus");
			getContextualMenus(contextMenuInfos, onCreateContextInfos);
		}catch(Exception e){
			e.printStackTrace();
			logger.error("Contextual: " + e);
		}
		try{
			logger.debug("Getting popup menus");
			getPopupMenus(popupInfos);
		}catch(Exception e){
			e.printStackTrace();
			logger.error("Popup: " + e);
		}
		try{
			getNavDropDownMenus(dropDownInfos);
		}catch(Exception e){
			e.printStackTrace();
			logger.error("NavDropDown: " + e);
		}
	}
	
	private void getNavDropDownMenus(Set<DropDownNavMenuInfo> dropDownInfos) {
		processDropDownNavMenuResults(dropDownInfos);
	}

	private void processDropDownNavMenuResults(Set<DropDownNavMenuInfo> dropDownInfos) {
		
		for (DropDownNavMenuInfo info : dropDownInfos){
			// setNavigationMode is not checked (see StmtSwitch for more explanation)
//			if (info.isSetActionBarThere() && info.isSetNavigationMode()){
			
			if (info.isSetActionBarThere()){
				
				// create new Menu
				Menu menu = new Menu(Content.getInstance().getNewUniqueID());
				menu.setKindOfMenu("NavDropDown");
				if (info.getListener() != null){
					menu.addListeners(info.getListener());
				}
				
				// create new ID for this UiElement
				String id = Integer.toString(content.getNewUniqueID());
				AppsUIElement uiE = new AppsUIElement("List", new ArrayList<Integer>(), id, new HashMap<>(), "", null, null, null);

				uiE.addText(info.getText(), "default_value");
										
				menu.addUIElement(uiE.getId());
				menu.addAssignedActivity(info.getActivityName());
				appController.addNewNavigationDropDownMenu(menu, uiE);
				appController.addXMLLayoutFileToActivity(info.getActivityName(), menu.getId());
			}
		}
	}

	private void getPopupMenus(Set<PopupMenuInfo> popupInfos) {
		processPopupMenuResults(popupInfos);
	}

	private void processPopupMenuResults(Set<PopupMenuInfo> popupInfos) {
		
		for (PopupMenuInfo info: popupInfos){
			// check if it is a popup menu and not e.g. an Optionsmenu 
				// other menu types call also the start point of the popup analysis
			if (info.isPopupMenu()){
				if ((info.getShowingItemReg() != null) && info.getShowingItemReg().equals("Item is passed in parameters")){
					
				}else if (!info.getShowingItemID().equals("")){
					
					// change the PopupMenu Layout XMLLayoutFile to Menu
					int layoutID = Integer.parseInt(info.getLayoutID());
					appController.extendXMLLayoutFileToMenu(layoutID);
					// set the popup activity as listener class to the popup listeners
					// NO: the stmtSwitch for popup menus searches the listener classes
//					for (Listener l : info.getListenerWRegs().values()){
//						l.setListenerClass(info.getActivityClassName());
//					}
					appController.addDataToMenuObject(layoutID, "Popup", info.getListenerWRegs().values());

					// add the Menu to the AppsUiElement where it is displayed
					int elementID = Integer.parseInt(info.getShowingItemID());
					appController.extendAppsUIElementWithSpecialTag(elementID);
					appController.addXMLLayoutFileToSpecialXMLTag(elementID, layoutID);//					}
				}else{
					// something went wrong
					logger.error("Is popupMenu, but no ItemID or ItemReg");
				}
			}
		}
	}

	private void getContextualMenus(Set<MenuInfo> contextMenuInfos, Set<MenuInfo> onCreateContextInfos){
		processContextualMenuResults(onCreateContextInfos, contextMenuInfos);
	}
	
	private void processContextualMenuResults(Set<MenuInfo> onCreateContextInfos,
			Set<MenuInfo> contextMenuInfos) {
		
		for (MenuInfo onCreateInfo: onCreateContextInfos){
			logger.debug("Contextual element info {}", onCreateInfo);
			
			for (MenuInfo contextMenuInfo : contextMenuInfos){
				try{ //TODO, for every subclass
					if (onCreateInfo.getActivityClassName().equals(contextMenuInfo.getActivityClassName())){
						logger.debug("Contextual Menu info {}", contextMenuInfo);
						// create Menu object out of the XMLLayoutFile with the LayoutID of the ContextMenu
						int menuID = Integer.parseInt(contextMenuInfo.getLayoutID());
						if(contextMenuInfo.isXMLDeclared()){
							appController.extendXMLLayoutFileToMenu(menuID);
							
						}
						else{ //ADDED
							//create a fake xml layout file for the menu ?
							appController.extendPhantomXMLFileForDynMenu(menuID, contextMenuInfo);
						}
						
						//TODO: uncomment when adding context menus
						appController.addXMLLayoutFileToActivity(contextMenuInfo.getActivityClassName(), menuID);

						// check if listener are existing for this Menu
						boolean dynListener = CheckIfMethodsExisting.getInstance().isMethodExisting(onCreateInfo.getActivityClassName(),
								"boolean onContextItemSelected(android.view.MenuItem)");

						// add the listener to an Collection<Listener> because the appController method wants it
						Collection<Listener> listeners = new HashSet<Listener>();
						Listener listener = null;
						if (dynListener){
							listener = new Listener("onLongClick", false, "boolean onContextItemSelected(android.view.MenuItem)", contextMenuInfo.getDeclaringSootClass());
							listener.setListenerClass(onCreateInfo.getActivityClassName());
							listeners.add(listener);
						}
						else {
							dynListener = CheckIfMethodsExisting.getInstance().isMethodExisting(onCreateInfo.getDeclaringSootClass(),
									"boolean onMenuItemSelected(int,android.view.MenuItem)");
							if (dynListener){
								listener = new Listener("onLongClick", false, "boolean onContextItemSelected(android.view.MenuItem)", contextMenuInfo.getDeclaringSootClass());
								listener.setListenerClass(onCreateInfo.getActivityClassName());
								listeners.add(listener);
							}
						}

						appController.addDataToMenuObject(menuID, "Contextual", listeners);

						// replace the AppsUIElement where the ContextualMenu is shown, to a SpecialTag and add the Menu to it
						String layoutID = onCreateInfo.getLayoutID();
						//TODO: Deal with this
						if(!layoutID.isEmpty()){
							int elementID = Integer.parseInt(layoutID);
							appController.extendAppsUIElementWithSpecialTag(elementID);
							appController.addXMLLayoutFileToSpecialXMLTag(elementID, menuID);
						}
						else
							logger.error("Empty element ID for context menu {} onCreate info {}, needs debugging", contextMenuInfo, onCreateInfo);
						
	//					// replace the AppsUIElement where the ContextualMenu is shown, to a SpecialTag and add the Menu to it
	//					for(XMLLayoutFile xmlF : app.getAllXMLLayoutFiles()){
	//						if (xmlF.containsAppsUIElementWithoutIncludings(onCreateInfo.getLayoutID())){
	//							SpecialXMLTag spec = xmlF.extendAppsUIElementWithSpecialTag(onCreateInfo.getLayoutID());
	//							spec.addXmlFiles(menu);
	//							if (dynListener)
	//								spec.addListener(listener);
	//							break;
	//						}
	//
	//					}
					}
				} catch(IllegalArgumentException e){
					logger.error("SearchContextMenus:" + "XMLLayoutFile file with this ID was not found: " + contextMenuInfo.getLayoutID());
				}
			}
		}
	}

	private void getOptionMenus(Set<MenuInfo> optionInfos){
		processOptionMenuResults(optionInfos);
	}
	
	private void processOptionMenuResults(Set<MenuInfo> menuInfos){
		for (MenuInfo mi: menuInfos){ //TODO parse subclasses, we'll have to also make sure the transitions are built for those subclasses evn if the callback is only parsed for the super class?
			logger.debug("Option Menu info {}", mi);
			try{
				int menuId = Integer.parseInt(mi.getLayoutID());
				if(mi.isXMLDeclared()){
					appController.extendXMLLayoutFileToMenu(menuId);
					appController.addXMLLayoutFileToActivity(mi.getActivityClassName(), menuId);
				}
				else{
					//create a fake xml layout file for the menu ?
					appController.extendPhantomXMLFileForDynMenu(menuId, mi);
					appController.addXMLLayoutFileToActivity(mi.getActivityClassName(), menuId);
				}

				//Here go through all subclasses which do not override onOptionsItemSelected?

				// check if a layout listener is attached
				boolean dynListener = CheckIfMethodsExisting.getInstance().isMethodExisting(mi.getDeclaringSootClass(),
						"boolean onOptionsItemSelected(android.view.MenuItem)");
				Listener listener = null;
				if (dynListener){
					listener = new Listener("onClick", false, "boolean onOptionsItemSelected(android.view.MenuItem)", mi.getDeclaringSootClass());
					listener.setListenerClass(mi.getActivityClassName());
				}
				else{
					dynListener = CheckIfMethodsExisting.getInstance().isMethodExisting(mi.getDeclaringSootClass(),
						"boolean onMenuItemSelected(int,android.view.MenuItem)");
					if (dynListener){
						listener = new Listener("onClick", false, "boolean onMenuItemSelected(int,android.view.MenuItem)", mi.getDeclaringSootClass());
						listener.setListenerClass(mi.getActivityClassName());
					}
					//NEED TO ADD THAT TO LISTENERINFO? BUT HOW TO MAP THE UI ELEMENTS THO lol
					//else dynListener = CheckIfMethodsExisting.getInstance().isMethodExisting(mi.getDeclaringSootClass(), "boolean onMenuItemClick(android.view.MenuItem)" {
				}

				// add the listener to an Collection<Listener> because the appController method wants it
				Collection<Listener> listeners = new HashSet<Listener>();
				if(listener != null)
					listeners.add(listener);

				appController.addDataToMenuObject(menuId, "Option", listeners);

			} catch(IllegalArgumentException e){
				logger.error("SearchOptionMenus:" + "XMLLayoutFile file with this ID was not found: " + mi.getLayoutID());
			}
			catch (NullPointerException e) {
				logger.error("AppController:addDataToMenuObject: XMLLayoutFile not found: " + mi);
				Helper.saveToStatisticalFile("AppController:addDataToMenuObject: XMLLayoutFile not found with id: " + mi);
			}
		}
	}


	
	// for a datastructure with Screen (with the search of Layouts/Screens)
//	private Application processOptionMenuResults(Application app, List<MenuInfo> menuInfos, String pathToAppOutFolder, String androidSDK, String apkFilePath){
//		for (MenuInfo mi: menuInfos){
//			boolean found = false;
//			// add this Menu to the Screen where it is displayed
//			for (Screen sc : app.getScreens()){
//				if(mi.getActivityClassName().equals(sc.getActivityName())){
//					
//					try{
//						Menu menu = app.expandXMLLayoutFileWithMenu(mi.getLayoutID());
//						menu.setKindOfMenu("Options");
//
//						boolean dynListener = CheckIfMethodsExisting.newInstance().isMethodExisting(mi.getActivityClassName(),
//								"public boolean onOptionsItemSelected(android.view.MenuItem)", 
//								pathToAppOutFolder, androidSDK, apkFilePath);
//						if (dynListener){
//							Listener listener = new Listener("onClick", false, "boolean onOptionsItemSelected(android.view.MenuItem)");
//							listener.setListenerClass(mi.getActivityClassName());
//							menu.addListeners(listener);
//						}
//						sc.addXmlFiles(menu);
//						found = true;
//						break;
//					}catch(IllegalArgumentException e){
//						logger.error(app.getName() + ": " + "XMLLayoutFile file with this ID was not found: " + mi.getLayoutID());
//					}
//				}
//			}
//			if (found)
//				continue;
//		}
//		return app;
//	}
	
	
	// for a datastructure with Screen (with the search of Layouts/Screens)
//	private Application processDropDownNavMenuResults(Application app,String outPath, List<DropDownNavMenuInfo> dropDownInfos,String androidSDK, String apkFilePath) {
//		
//		for (DropDownNavMenuInfo info : dropDownInfos){
//			if (info.isSetActionBarThere() && info.isSetNavigationMode()){
//				
//				for (Screen sc : app.getScreens()){
//					if (sc.getActivityName().equals(info.getActivityName())){
//						// create new Menu
//						Menu menu = new Menu();
//						menu.setKindOfMenu("NavDropDown");
//						if (info.getListener() != null){
//							menu.addListeners(info.getListener());
//						}
//						menu.addAssignedActivity(info.getActivityName());
//						
//						AppsUIElement uiE = new AppsUIElement("List", null, null, null, null);
//						uiE.setText(info.getTextFromElement());
//												
//						menu.addUIElement(uiE);	
//						menu.setName("-NewFoundInCode-");
//						app.addXMLLayoutFile(menu);
//						sc.addXmlFiles(menu);
//						break;
//					}
//				}
//			}
//		}
//		
//		return app;
//	}
	
}
