package GitHubSample;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.URL;
import java.net.URLConnection;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.NodeList;

import aim_One.ProcessPMIDXML;

//import java.sql.Connection;

public class HighImpactPubMedArticleRanking {



	// this version is used to generate the same PMIDs list based on 
	// the relevance sort, so that I can generate a baseline for my study 
	// on June/04/2016. reference for relevance sort:
	// http://www.ncbi.nlm.nih.gov/news/03-10-2014-eutils-sorting/
	// https://www.nlm.nih.gov/pubs/techbull/so13/so13_pm_relevance.html
	
	public final static String eUtilHeader = "https://eutils.ncbi.nlm.nih.gov/"
			+ "entrez/eutils/esearch.fcgi?db=pubmed&sort=relevance&term=";
	
////this version is the original version without relevance sort	
//	public final static String eUtilHeader = "http://eutils.ncbi.nlm.nih.gov/"
//	+ "entrez/eutils/esearch.fcgi?db=pubmed&term=";
	
	
	// "Cardiomyopathy, Hypertrophic"[MeSH Terms]
	// public static String diseaseTerm = "\"" + "Cardiomyopathy, Hypertrophic"
	// + "\"[MeSH Terms] ";

	
	/*
	 * (Therapy/Narrow[filter] OR ("therapy"[Subheading] AND systematic[sb] AND
	 * ("systematic review"[ti] OR "meta-analysis"[ti] OR
	 * "Cochrane Database Syst Rev"[journal])))
	 */
	public final static String RCTSRNarrowStudies = "AND ("
			+ "Therapy/Narrow[filter] "
			+ "OR (\"therapy\"[Subheading] AND systematic[sb] "
			+ "AND (\"systematic review\"[ti] OR \"meta-analysis\"[ti] "
			+ "OR \"Cochrane Database Syst Rev\"[journal]))" + ")";
	
	
	
	/*
	 * (Therapy/Broad[filter] OR ("therapy"[Subheading] AND systematic[sb] AND
	 * ("systematic review"[ti] OR "meta-analysis"[ti] OR
	 * "Cochrane Database Syst Rev"[journal])))
	 */
	public final static String RCTSRBroadStudies = "AND ("
			+ "Therapy/Broad[filter] "
			+ "OR (\"therapy\"[Subheading] AND systematic[sb] "
			+ "AND (\"systematic review\"[ti] OR \"meta-analysis\"[ti] "
			+ "OR \"Cochrane Database Syst Rev\"[journal]))" + ")";

	/*
	 * ( Therapy/Broad[filter] OR ("therapy"[Subheading] AND systematic[sb] AND
	 * ("systematic review"[ti] OR "meta-analysis"[ti] OR
	 * "Cochrane Database Syst Rev"[journal])) OR ("therapy"[Subheading] AND
	 * ("Support of Research" [Publication Type] OR
	 * "Comparative Study" [Publication Type] OR
	 * "Multicenter Study"[Publication Type] OR "Meta-Analysis"[Publication
	 * Type]) ) )
	 */

	public final static String RCTSRBroadPLUSStudies = "AND ("
			+ "Therapy/Broad[filter] "
			+ "OR (\"therapy\"[Subheading] AND systematic[sb] "
			+ "AND (\"systematic review\"[ti] OR \"meta-analysis\"[ti] "
			+ "OR \"Cochrane Database Syst Rev\"[journal]))"
			+ "OR (\"therapy\"[Subheading] AND (\"Comparative Study\"[Publication Type]"
			+ "OR \"Support of Research\"[Publication Type]"
			+ "OR \"Multicenter Study\"[Publication Type]"
			+ "OR \"Meta-Analysis\"[Publication Type])" + "))";

	/*
	 * AND "humans"[MeSH Terms] AND "english"[language] AND hasabstract[text]
	 */
	public final static String furtherContraints = "AND \"humans\"[MeSH Terms] AND \"english\"[language] AND hasabstract[text]";

	/*
	 * &retmax=100000 &usehistory=y
	 */
	public final static String eUtilTail = "&retmax=100000&usehistory=y";

	private static LinkedList<String> tmpll = new LinkedList<String>();
	
	

	public HighImpactPubMedArticleRanking() {

	}

	public static void main(String[] args) throws Exception {
		// System.out.println("Start!");

		Date d = new Date();
		System.out
				.println("==================================================================================================================");
		System.out.println(d.toString());
		System.out.println();
		
		
		
//		System.out.println("Comorbidity_Test:");
//		getComorbidityForMultiplePMIDWithInputDisease("Comorbidity_Test.txt");
//		getComorbidityForMultiplePMIDWithInputDisease("Diseases_Dates_Summary.txt");

		
//		System.out.println("This try is for not containing any study type restrictions: ");
// 		getTopMeSHTermsWithSubheading();

//		getOneDiseaseGSRecallPercentages("Depressive Disorder, Major","2000/07/01","2006/12/31","MDD");
// 		getOneDiseaseGSRecallPercentages("Depressive Disorder","2000/07/01","2006/12/31","MDD");

		
		System.out.println("diseases_GS_recall_percentages:");
		System.out.println();
		
		getAllDiseasesGSRecallPercentages("Diseases_Dates_Summary.txt");
		
//		System.out.println(getPubMedQueryFromDiseaseWithDatesRange(
//				"Carotid Artery Diseases","2004/01/01","2010/05/30"));
		
//		System.out.println("diseases_GS_recall_percentages:");
//		System.out.println();
//		getAllDiseasesGSRecallPercentages("Diseases_Dates_Summary_With_Comorbidity.txt");
		
		
//		System.out.println("Diseases_Dates_Summary_OneLevelUP");
//		System.out.println();
//		getAllDiseasesGSRecallPercentages("Diseases_Dates_Summary_OneLevelUP.txt");
		
		
//		System.out.println("Diseases_Dates_Summary_MeSH_From_Guideline_Directly:");
//		System.out.println();
//		getAllDiseasesGSRecallPercentages("Diseases_Dates_Summary_MeSH_From_Guideline_Directly.txt");

//		System.out.println("CarotidArteryDisease_Stroke_False_Negative_Analysis:");
//		System.out.println();
//		getAllDiseasesGSRecallPercentages("CarotidArteryDisease_Stroke_False_Negative_Analysis.txt");
		
				
//		getPubTypeNMeSHOfFalseNegative("Diseases_Dates_Summary.txt");
//		getPubTypeNMeSHOfFalseNegative("CarotidArteryDisease_Stroke_False_Negative_Analysis.txt");

// 		getLinkedListFromAFile("50 Diseases.txt");

// 		getPMIDsFromDisease();

		System.out.println();
		System.out
				.println("==================================================================================================================");
		// System.out.println("Done!");

	}

	// to get the time constraints for the specific guidelines
	public static String getTimeConstraints(String s1, String s2) {
		return "AND (\"" + s1 + "\"[PDAT] : \" " + s2 + "\"[PDAT])";
	}

	public static String getPubMedQueryFromDiseaseWithDatesRange(
			String diseaseMeSH, String time1, String time2) {
		
		return

		getMultidiseasesMeSH(processInputDiseases(diseaseMeSH))
// 		getADiseaseMeSH(diseaseMeSH)
		
//		getMultidiseasesMeSHAllFieldsIndividually(processInputDiseases(diseaseMeSH))
//		getADiseaseMeSHAllFields(diseaseMeSH)
//		getADiseaseMeSHAllFieldsIndividually(diseaseMeSH)

		+ RCTSRNarrowStudies
		
//		+ RCTSRBroadStudies

//		+ RCTSRBroadPLUSStudies

		+ getTimeConstraints(time1, time2) 
		+ furtherContraints
		;
	}

	public static String getQueryFromDisease(String diseaseMeSH) {
		return getADiseaseMeSH(diseaseMeSH) + RCTSRBroadStudies
				+ getTimeConstraints("2004/01/01", "2015/01/01")
				+ furtherContraints;
	}

	// I may need to reconsider this

	// this method will be used to get multiple disease MeSH terms
	// if the starting disease MeSH term needs to be expanded
	public static String getMultidiseasesMeSH(
			LinkedList<String> listOfDiseaseMeSH) {

		String str = getADiseaseMeSH(listOfDiseaseMeSH.getFirst());

		if (listOfDiseaseMeSH.size() == 1) {
			return str;
		}

		else
			for (int i = 1; i < listOfDiseaseMeSH.size(); i++) {
				str = str + "OR"
						+ getADiseaseMeSH(listOfDiseaseMeSH.get(i));
			}
		return str;
	}

	
	// this method will be used to get multiple disease MeSH terms with all fields property inclusion
	// if the starting disease MeSH term needs to be expanded
	public static String getMultidiseasesMeSHAllFieldsIndividually(
			LinkedList<String> listOfDiseaseMeSH) {

		String str = getADiseaseMeSHAllFieldsIndividually(listOfDiseaseMeSH.getFirst());

		if (listOfDiseaseMeSH.size() == 1) {
			return str;
		}

		else
			for (int i = 1; i < listOfDiseaseMeSH.size(); i++) {
				str = str + "OR"
						+ getADiseaseMeSHAllFieldsIndividually(listOfDiseaseMeSH.get(i));
			}
		return str;
	}
	
	
	// This method is used for taking care of starting the PubMed Query
	// with multiple diseases (I think single disease should be still fine)
	public static LinkedList<String> processInputDiseases(String diseases) {
		LinkedList<String> diseasesLL = new LinkedList<String>();
		String[] diseasesArr = diseases.split(";");
		for (String str : diseasesArr) {
			diseasesLL.add(str);
		}
		return diseasesLL;
	}

	public static String getADiseaseMeSH(String diseaseMeSH) {
		return "\"" + diseaseMeSH + "\"[MeSH Terms] ";
	}

	public static String getADiseaseMeSHAllFields(String diseaseMeSH) {
		return "(" + "\"" + diseaseMeSH + "\"[MeSH Terms] " + "OR" + "\""
				+ diseaseMeSH + "\"[All Fields] " + ")";
	}

	// expand the search to wild search in PubMed
	public static String getADiseaseMeSHAllFieldsIndividually(
			String diseaseMeSH) {
		return "(" + "\"" + diseaseMeSH + "\"[MeSH Terms] " + "OR"
				+ getADiseaseMeSHAllFieldsIndividuallyHelper(diseaseMeSH)
				+ "OR" + "\"" + diseaseMeSH + "\"[All Fields] " + ")";
	}

	// This is a helper method to get the method
	// getdiseaseMeSH_AllFields_Individually
	public static String getADiseaseMeSHAllFieldsIndividuallyHelper(
			String diseaseMeSH) {
		String str = "";
		// Use the regex "\W" to match any non-word character.
		// In this particular case, space and comma apply.
		String[] tokens = diseaseMeSH.split("\\W");

		if (tokens.length == 0) {
			return "\"" + tokens[0] + "\"[All Fields] ";
		}

		else {
			str = "\"" + tokens[0] + "\"[All Fields] ";

			for (int i = 1; i < tokens.length; i++) {
				str = str + "AND" + "\"" + tokens[i] + "\"[All Fields] ";
			}

			return "(" + str + ")";
		}

	}

	// replace the special characters in the regular query
	// with the characters/string used in eUtil

	public static String getUtilURL(String str) {
		return eUtilHeader
				+ str.replace(" ", "+").replace(",", "%2C")
						.replace("\"", "%22").replace("(", "%28")
						.replace("/", "%2F").replace(")", "%29")
						.replace(":", "%3A") + eUtilTail;
	}

	// getPMIDsFromDiseaseDates
	private static LinkedList<String> getPMIDsFromEUtilForDiseaseWithDatesRange(
			String disease, String d1, String d2) throws Exception {
		URL eUtilURL = new URL(
				getUtilURL(getPubMedQueryFromDiseaseWithDatesRange(
						disease, d1, d2)));

//		System.out.println(getPubMedQueryFromDiseaseWithDatesRange(
//						disease, d1, d2));
		
		try {

			URLConnection connection = eUtilURL.openConnection();

			Document doc = parseXML(connection.getInputStream());
			NodeList PMIDNodes = doc.getElementsByTagName("Id");

			tmpll.clear();

			for (int i = 0; i < PMIDNodes.getLength(); i++) {
				tmpll.add(PMIDNodes.item(i).getTextContent());
				// System.out.println(PMIDNodes.item(i).getTextContent());
			}

			return tmpll;

		} catch (IOException e) {
			throw e;
		}

	}
	
//
//	private static void getPMIDsFromDisease() throws Exception {
//		URL eUtilURL = new URL(
//				getUtilURL(getQueryFromDisease("Myocardial Infarction")));
//
//		try {
//
//			URLConnection connection = eUtilURL.openConnection();
//
//			Document doc = parseXML(connection.getInputStream());
//			NodeList PMIDNodes = doc.getElementsByTagName("Id");
//
//			System.out.println(PMIDNodes.getLength());
//
//			/*
//			 * for (int i = 0; i < descNodes.getLength(); i++) {
//			 * System.out.println(PMIDNodes.item(i).getTextContent()); }
//			 */
//
//		} catch (IOException e) {
//			throw e;
//		}
//
//	}

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

	// findAndProcessFileInAFolder,
	// return a the item in the txt file as a LinkedList
	public static LinkedList<String> findAndProcessFileInAFolder(
			final File folder, String str) throws Exception {
		LinkedList<String> ll = new LinkedList<String>();
		try {
			for (final File fileEntry : folder.listFiles()) {
				String file_name = fileEntry.getName().substring(0,
						fileEntry.getName().length() - 4);

				if (file_name.equalsIgnoreCase(str)) {
					// use the whole directory to find the specific disease GS citations
					
					
//					ll = getLinkedListFromAFile("E:/Studying/eclipse/workspace/Thesis/RCTs_N_SRs_PMIDs/"
					ll = getLinkedListFromAFile("E:/Studying/Box Sync/workspace/Thesis/RCTs_N_SRs_PMIDs/"

//					ll = getLinkedListFromAFile("E:/Studying/eclipse/workspace/Thesis/PMIDs/"
//					ll = getLinkedListFromAFile("E:/Studying/Box Sync/workspace/Thesis/PMIDs/"
							+ file_name + ".txt");
				}
			}

		} catch (Exception ex) {
			throw ex;
		}
		return ll;
	}

	// this is a method used to compare whether any search
	public static void getTopMeSHTermsWithSubheading() throws Exception {

		LinkedList<String> GSLL = findAndProcessFileInAFolder(new File(				
//				"E:/Studying/eclipse/workspace/Thesis/RCTs_N_SRs_PMIDs"), "2011.hc");
				"E:/Studying/Box Sync/workspace/Thesis/RCTs_N_SRs_PMIDs"), "2011.hc");
		
//				"E:/Studying/eclipse/workspace/Thesis/PMIDs"), "2011.hc");
//		        "E:/Studying/Box Sync/workspace/Thesis/PMIDs"), "2011.hc");
		System.out.println("2011.hc GS results:");

		HashMap<String, Integer> countTopMeSHTermsWithSubheadingGS = new HashMap<String, Integer>();
		for (String str : GSLL) {
			LinkedList<String> ll = ProcessPMIDXML
					.getOnlyTheMeSHTermsWithSubheadingOfAPMID(str);
			for (String str1 : ll) {
				countTopMeSHTermsWithSubheadingGS = AllGSPubTypeAnalysis
						.countPubType(countTopMeSHTermsWithSubheadingGS,
								str1);
			}
		}

		AllGSPubTypeAnalysis
				.printAMap(countTopMeSHTermsWithSubheadingGS);

		LinkedList<String> queryResultLL = getPMIDsFromEUtilForDiseaseWithDatesRange(
				"Cardiomyopathy, Hypertrophic", "2004/01/01", "2011/01/20");
		System.out.println("2011.hc Query first 100 results:");

		HashMap<String, Integer> countTopMeSHTermsWithSubheadingQuery = new HashMap<String, Integer>();
		for (int i = 0; i < 100; i++) {
			LinkedList<String> ll = ProcessPMIDXML
					.getOnlyTheMeSHTermsWithSubheadingOfAPMID(queryResultLL
							.get(i));
			for (String str1 : ll) {
				countTopMeSHTermsWithSubheadingQuery = AllGSPubTypeAnalysis
						.countPubType(
								countTopMeSHTermsWithSubheadingQuery, str1);
			}
		}

		AllGSPubTypeAnalysis
				.printAMap(countTopMeSHTermsWithSubheadingQuery);

	}

	public static LinkedList<String> getLinkedListFromAFile(String str) throws Exception {

		LinkedList<String> ll = new LinkedList<String>();

		try {

			File f = new File(str);
			BufferedReader bufRead = new BufferedReader(new InputStreamReader(
					new FileInputStream(f)));
			String S1 = "";

// 			This is the version I only want to include all the GS citations 
// 			with certain pubtype listed below			
//			while ((S1 = bufRead.readLine()) != null) {
//				
//				if (Process_PMID_XML.check_PubType_For_A_PMID(S1,
//						"Clinical Trial", "Review", "Meta-Analysis")
//				// || Process_PMID_XML.check_PubType_For_A_PMID(S1, "Review")
//				// || Process_PMID_XML.check_PubType_For_A_PMID(S1,
//				// "Meta-Analysis")
//				) {
//					System.out.println(S1);
//					ll.add(S1);
//				}
//			}
			
// 			this is the version I want to include all the GS citations 
//			regardless of its indexed Publication types			
			while ((S1 = bufRead.readLine()) != null) {
				// System.out.println(S1);
				ll.add(S1);
			}


			bufRead.close();
		} catch (IOException ioe) {
			System.out.println("Something is wrong with reading file! "
					+ "Can't find the appropriate file!");
		}

		return ll;

	}

	public static void getComorbidityForMultiplePMIDWithInputDisease(
			String diseases_List) throws Exception {
		
		try {
			File f = new File(diseases_List);
			BufferedReader bufRead = new BufferedReader(new InputStreamReader(
					new FileInputStream(f)));
			String S1 = "";
			while ((S1 = bufRead.readLine()) != null) {
				String[] aa = S1.split("::");
				LinkedList<String> queryResultLL = getPMIDsFromEUtilForDiseaseWithDatesRange(
						aa[0], aa[1], aa[2]);
				System.out.println();
				System.out.println("=============> " + aa[0] + " <=============");
				for (int i = 0; i < 200; i++) {
					ProcessPMIDXML
							.get_the_comorbidity_for_A_PMID_with_input_disease(
									aa[0], queryResultLL.get(i));
				}
			}
			bufRead.close();
		} catch (IOException ioe) {
			System.out.println("Something is wrong with reading file! "
					+ "Can't find the appropriate file!");
		}
	}

	
	
	public static void getAllDiseasesGSRecallPercentages(
			String diseases_List) throws Exception {
		System.out.printf("%-40s %-15s %-15s %-15s %-15s", "Guideline",
				"Query_Results", "Intersection", "GS_citations", "GS_recall");
		System.out.println();
		try {

			File f = new File(diseases_List);
			BufferedReader bufRead = new BufferedReader(new InputStreamReader(
					new FileInputStream(f)));
			String S1 = "";
			while ((S1 = bufRead.readLine()) != null) {
				String[] aa = S1.split("::");
				getOneDiseaseGSRecallPercentages(aa[0], aa[1], aa[2],
						aa[3]);
			}

			bufRead.close();
		} catch (IOException ioe) {
			System.out.println("Something is wrong with reading file! "
					+ "Can't find the appropriate file!");
		}
	}

	public static void getOneDiseaseGSRecallPercentages(
			String diseases_MeSH_terms, String start_date, String end_date,
			String guideline_name) throws Exception {

		LinkedList<String> GSLL = findAndProcessFileInAFolder(new File(				
//				"E:/Studying/eclipse/workspace/Thesis/RCTs_N_SRs_PMIDs"), guideline_name);
//				"E:/Studying/eclipse/workspace/Thesis/PMIDs"), guideline_name);
		
				"E:/Studying/Box Sync/workspace/Thesis/RCTs_N_SRs_PMIDs"), guideline_name);
//				"E:/Studying/Box Sync/workspace/Thesis/PMIDs"), guideline_name);
		
		LinkedList<String> queryResultLL = getPMIDsFromEUtilForDiseaseWithDatesRange(
				diseases_MeSH_terms, start_date, end_date);
		

//		working with database, usually, this part should be commented out 
//		unless I know what need to work with the database		
//		If I don't use this part, it will be complained that "too many connections"

//		java.sql.Connection conn = MySQL.SemanticMedlineConnection();	
		
				
//		
// 		this part is used to populate MySQL Database, 
//		please comment out if we don't need to touch database
 
//		System.out.println(guideline_name);
		PrintWriter writer = new PrintWriter(
				
//				"E:/Studying/eclipse/workspace/Thesis/DataSet_PMIDs/"
				"E:/Studying/Box Sync/workspace/Thesis/DataSet_PMIDs/"
						+ guideline_name + ".txt", "UTF-8");
		for (String PMID : queryResultLL) {
			writer.println(PMID);
			
//		 working with database, usually, this part should be commented out unless I know
//		 what need to work with the database		 
	
//			MySQL.insert_A_Record_Into_Table("DataSet", PMID, guideline_name, conn);	
		
		}
		
		writer.close();
	
//		*/

		int queryResultLLOriSize = queryResultLL.size();
		
		queryResultLL.retainAll(GSLL);
	
		double recall = queryResultLL.size() / (double) GSLL.size();

		System.out.printf("%-40s %-15s %-15s %-15s %.2f%%",
				diseases_MeSH_terms, queryResultLLOriSize,
				queryResultLL.size(), GSLL.size(), recall * 100);
		System.out.println();

	}

	public static void getPubTypeNMeSHOfFalseNegative(
			String diseases_List) throws Exception {
		ProcessPMIDXML pubTypeMeSH = new ProcessPMIDXML();
		try {

			File f = new File(diseases_List);
			BufferedReader bufRead = new BufferedReader(new InputStreamReader(
					new FileInputStream(f)));
			String S1 = "";
			while ((S1 = bufRead.readLine()) != null) {
				String[] aa = S1.split("::");

				LinkedList<String> GSLL = findAndProcessFileInAFolder(
						
//						new File("E:/Studying/eclipse/workspace/Thesis/RCTs_N_SRs_PMIDs"),
//						new File("E:/Studying/eclipse/workspace/Thesis/PMIDs"),
						
						new File("E:/Studying/Box Sync/workspace/Thesis/RCTs_N_SRs_PMIDs"),
//						new File("E:/Studying/Box Sync/workspace/Thesis/PMIDs"),
						
						
						aa[3]);
				LinkedList<String> queryResultLL = getPMIDsFromEUtilForDiseaseWithDatesRange(
						aa[0], aa[1], aa[2]);

				GSLL.removeAll(queryResultLL);
				System.out.println(aa[0]);

				PrintWriter writer = new PrintWriter(
//						"E:/Studying/eclipse/workspace/Thesis/Results/" + aa[0]	+ ".txt", "UTF-8");
						"E:/Studying/Box Sync/workspace/Thesis/Results/" + aa[0]	+ ".txt", "UTF-8");

				for (String PMID : GSLL) {
					writer.println(pubTypeMeSH
							.getPubTypesAndMeSHTermsOfAPMIDStringVersionForFile(PMID));
				}

				writer.close();
				System.out.println(aa[0] + " is done.\n");
			}

			bufRead.close();
		} catch (IOException ioe) {
			System.out.println("Something is wrong with reading file! "
					+ "Can't find the appropriate file!");
		}
	}

}
