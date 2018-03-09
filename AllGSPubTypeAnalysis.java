package GitHubSample;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.URLConnection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map.Entry;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.NodeList;



public class AllGSPubTypeAnalysis {

	private static HashMap<String, Integer> pubTypeCount = new HashMap<String, Integer>();
	
	public static void main(String[] args) throws Exception {
		System.out.println("Start!");
		System.out.println("===================================");
		System.out.println();
		
//		pubTypeCount.clear();		
//		goThroughFilesForFolder(new File(
////				"E:/Studying/eclipse/workspace/Thesis/PMIDs"
//				"E:/Studying/Box Sync/workspace/Thesis/PMIDs"
//				));
//		

		
		// A very interesting discussion about why the original codes does't
		// work (although it did work before and I am not sure the reason)
		// http://stackoverflow.com/questions/24934777/null-pointer-exception-when-reading-textfile
//		ReadAllDiseases("E:/Studying/Box Sync/workspace/Thesis/PMIDs/Diseases_Dates_Summary.txt"); 
		
		/*
	    for (Entry<String, Integer> entry : pubTypeCount.entrySet()) {
	        String key = entry.getKey().toString();;
	        Integer value = entry.getValue();
	        System.out.println(key + ":  " + value );
	    }
		*/
	    System.out.println();
		System.out.println("===================================");
		System.out.println("Done!");

	}

	public static void printAMap(HashMap<String, Integer> HM){
	    for (Entry<String, Integer> entry : HM.entrySet()) {
	        String key = entry.getKey().toString();;
	        Integer value = entry.getValue();
	        System.out.println(key + ":  " + value );
	    }
	}
	
	// this method is used to count the publication types
	public static HashMap<String, Integer> countPubType(HashMap<String, Integer> m,
			String str) {
		if (!m.containsKey(str)) {
			m.put(str, 1);
		}
		
		else {
			m.put(str, m.get(str) + 1);
		}

		return m;
	}

	// this is a generic method to go through the folders
	// it is OK even the folder contains sub-folder
	public static void goThroughFilesForFolder(final File folder) throws Exception {

		/*
		 * char[] filePath;
		// *Files.walk(Paths.get("E:/Studying/eclipse/workspace/Thesis/PMIDs")).
		 * Files.walk(Paths.get("E:/Studying/Box Sync/workspace/Thesis/PMIDs")).
		 * forEach(filePath -> { if (Files.isRegularFile(filePath)) {
		 * System.out.println(filePath); } }
		 */
		// /*
		for (final File fileEntry : folder.listFiles()) {
			if (fileEntry.isDirectory()) {
				goThroughFilesForFolder(fileEntry);
			} else {
				processFile(fileEntry);
				// System.out.println(fileEntry.getName());
			}
		}
		// */
	}

	public static void processFile(File f) throws Exception {
		// LinkedList<String> allDiseases = new LinkedList<String>();
		try {
			BufferedReader bufRead = new BufferedReader(new InputStreamReader(
					new FileInputStream(f)));
			String S1 = "";
			while ((S1 = bufRead.readLine()) != null) {
				//System.out.println(S1);
				getPubTypes(S1);
				//System.out.println();
			}
			bufRead.close();
		} catch (IOException ioe) {
			System.out.println("Something is wrong with reading file! "
					+ "Can't find the appropriate file!");
		}

		// return allDiseases;

	}

	private static void getPubTypes(String str) throws Exception {
		URL eUtilURL = new URL(getUtilURL(str));
		// URL eUtilURL = new URL(getUtilURL("10665556"));
		try {
			URLConnection connection = eUtilURL.openConnection();
			Document doc = parseXML(connection.getInputStream());
			NodeList PubTypesNodes = doc
					.getElementsByTagName("PublicationType");
			for (int i = 0; i < PubTypesNodes.getLength(); i++) {
				// System.out.println(PubTypesNodes.item(i).getTextContent());

				countPubType(pubTypeCount, PubTypesNodes.item(i)
						.getTextContent());
			}

			/*
			 * txt = txt + "Publication Type".bold() + "<br>" ; for
			 * (i=0;i<p.length;i++) { txt=txt + p[i].childNodes[0].nodeValue +
			 * "<br>"; }
			 * txt=txt + "<br>";
			 */
			// System.out.println(PubTypesNodes.getLength());

			/*
			 * for (int i = 0; i < descNodes.getLength(); i++) {
			 * System.out.println(PMIDNodes.item(i).getTextContent()); }
			 */

		} catch (IOException e) {
			throw e;
		}

	}

	public static String getUtilURL(String PMID) {
		return "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id="
				+ PMID + "&retmode=xml";
	}

	private static Document parseXML(InputStream stream) throws Exception {
		DocumentBuilderFactory objDBF = null;
		DocumentBuilder objDB = null;
		Document doc = null;
		try {
			objDBF = DocumentBuilderFactory.newInstance();
			objDB = objDBF.newDocumentBuilder();

			doc = objDB.parse(stream);
		} catch (Exception ex) {
			throw ex;
		}

		return doc;
	}

	public static LinkedList<String> readAllDiseases(String str) {

		LinkedList<String> allDiseases = new LinkedList<String>();

		try {
			// File f = new File("AllSymbols.txt");
			// File f = new File("AllSymbolsGT0.3M.txt");
			// File f = new File("AllSymbols-New.txt");
			File f = new File(str);
			BufferedReader bufRead = new BufferedReader(new InputStreamReader(
					new FileInputStream(f)));
			String S1 = "";
			while ((S1 = bufRead.readLine()) != null) {
				System.out.println(S1);
				allDiseases.add(S1);
			}

			bufRead.close();
		} catch (IOException ioe) {
			System.out.println("Something is wrong with reading file! "
					+ "Can't find the appropriate file!");
		}

		return allDiseases;

	}

}
