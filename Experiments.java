package GitHubSample;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.LinkedList;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;


public class Experiments {

	// depends on how many guidelines will be included, can be 11 or 10 ...

	public static double[] precisionsArray = new double[10];
	// public static double[] precisionsArray = new double[11];

	private static String filesRootPath = "E:/Studying/Box Sync/Ph.D Projects"
			+ "/Thesis Aims/Aim 1 - Medline Articles Ranking" + "/Data and Analysis/Data/";

	private static String specificFolderPath = "1+1+9/";

	// private static String specificFolderPath = "1 + 10/";

//	private static String main_path = "E:/Studying/Box Sync/Ph.D Projects/Thesis Aims/"
//			+ "Aim 1 - Medline Articles Ranking/Data and Analysis/Amin "
//			+ "Hyperparameter Optimization/JavaCodes_ParameterCombinations";

	// private static String specificFolderPath = "6 + 5/";

	// private static String specificFolderPath = "11/";

	public static String[] attributes = { 
			"All_Attributes", 
//			"ScopusCC_Removed", 
//			"AltmetricScore_Removed",
//     		"AltmetricScore_N_ScopusCC_Removed",

     		// "Only_ScopusCC",
			// "Halil_Probability_Removed",
			// "JIF_Removed","Only_Metadata_Contained"
	};

	public static String[] guidelines = {
			// "hf2014",
			// "MDD",
			"ecvad2011", "sihd2012", "mi2013", "af2014", "vhd2014", "COPD2014", "RA", 
			"Asthma", 
			"COPD2007" 
			};

	public static String[] evaluations = {
			"topKPrecision", "topKMeanAveragePrecision", "topKMeanReciprocalRank", "Precision", "Recall", "F1", };

	public static String[][] weightedGuideline = { { "ecvad2011", "0.015", "0.000" }, { "sihd2012", "0.344", "0.000" },
			{ "mi2013", "0.163", "0.000" }, { "af2014", "0.082", "0.000" }, { "hf2014", "0.142", "0.000" },
			{ "vhd2014", "0.029", "0.000" }, { "COPD2014", "0.098", "0.000" }, { "RA", "0.023", "0.000" },
			{ "Asthma", "0.066", "0.000" }, { "COPD2007", "0.039", "0.000" } };


	public static LinkedList<Classifier> classifiers = new LinkedList<Classifier>();

	public static void main(String[] args) throws Exception {

		System.out.println("Start! \n");

//		// part of hypotheses testing to get baseline results
//		 getEvaluationsOnRelevanceSort(20);
		 
//		// part of hypotheses testing to get baseline results
//		 getEvaluationsOnHalilScore(20);

//		Step 4: use the chosen hyper parameter + cost matrix to work on
//		different attributes so that hypotheses can be tested
		 testHypothesesUsingTheTunedClassifier();

//		//Step 3: Set chosen hyper parameter for each classifier
//		// so that these classifiers can be used in cost matrix
//		testCostMatrix();

//		//Step 2: Set chosen hyper parameter for each classifier
//		// so that these classifiers can be used in cost matrix
//		 setChosenHyperParaForEachClassifier();

//		//Step 1: to get the hyper parameter for each classifier
//		// then identify a set of parameters for each classifier
//		 tryHyperParaCombinationsForEachClassifier();
		
		
		
		// System.out.println("");
		// System.out.println("");
		// System.out.println("Print all classifiers:");
		//
		// for (int i = 0; i < classifiers.size(); i++) {
		// System.out.println(classifiers.get(i).getClass());
		// trainClassifierOnOneDatasetThenTestOnAnotherDataset(classifiers.get(i),
		// "hf2014", "MDD");
		//// trainClassifierOnOneDatasetThenTestOnAnotherDataset(classifiers.get(i),
		// "hf2014", "MDD", i);
		// }
		//
		// classifiers.clear();

		System.out.println("done!");

	}

	// background about this part:
	// https://weka.wikispaces.com/Serialization
	// for the experiment version "1+1+9";
	public static double trainAClassifierOnADatasetThenTestOnAnotherDataset(Classifier model,
			String trainingGuidelineName, String testGuidelineName, String eval, 
			String attributesToBeRemoved) throws Exception {

		// public static double
		// trainClassifierOnOneDatasetThenTestOnAnotherDataset(Classifier
		// model,
		// String trainingGuidelineName, String testGuidelineName, int i)
		// throws Exception {

		// get the training data set, on which train the classifier
		// Instances originalTraining =
		// getTrainingOrTestInstancesForAGuideline(
		// guidelineName, "training");

		DataSource source;
		String filename = "";

		filename = filesRootPath + specificFolderPath + "ARFFs/" + trainingGuidelineName + ".arff";
		source = new DataSource(filename);
		Instances training = source.getDataSet();

		if (training.classIndex() == -1) {
			training.setClassIndex(training.numAttributes() - 1);
		}
		
		
		// used to remove certain attributes for hypothesis testing
		Instances attributesRemovedTraining = removeCertainAttributes(training, attributesToBeRemoved,
				trainingGuidelineName, "");
				
		
//		//// the datasets are already processed into Binary, no need the following
//		// need to do this transformation to make NaiveBayesMultinomial work
//		// meanwhile other classifiers still work
//		NominalToBinary filter = new NominalToBinary();	// setup filter
//		filter.setInputFormat(training);		
//		Instances training2 = Filter.useFilter(training, filter); // apply filter
//		model.buildClassifier(training2);
		

		model.buildClassifier(attributesRemovedTraining);
		
		
		// the path the model will be saved into
		String modelStoredPath = filesRootPath + specificFolderPath + "Models/" + model.getClass().getName();

		// System.out.println(modelStoredPath);

		// Thread.sleep(1000000);
		// serialize model
		weka.core.SerializationHelper.write(modelStoredPath, model);

		// de-serialize the classifier.
		Classifier classifier = (Classifier) weka.core.SerializationHelper.read(modelStoredPath);

		// load the test instance

		filename = filesRootPath + specificFolderPath + "ARFFs/" + testGuidelineName + ".arff";
		source = new DataSource(filename);
	
//		// //the data sets are already processed into Binary, no need the following
//		// need to do this transformation to make NaiveBayesMultinomial work
//		// meanwhile other classifiers still work
//		Instances testInstances1 = source.getDataSet();
//		Instances testInstances = Filter.useFilter(testInstances1, filter);
		
	
		Instances testInstances = source.getDataSet();


//		// Mark the last attribute in each instance as the true class.
//		if (testInstances.classIndex() == -1) {
//			testInstances.setClassIndex(testInstances.numAttributes() - 1);
//		}

		// // Load the test instances.
		// Instances originalTestInstances =
		// getTrainingOrTestInstancesForAGuideline(testGuidelineName,
		// "test");
		//
		// Instances attributesDeduplicatedTestInstances =
		// removeDuplicatedAttributes(originalTestInstances,
		// guidelineName, "test");
		//
		// Instances attributesRemovedTestInstances =
		// removeCertainAttributes(attributesDeduplicatedTestInstances,
		// attributesToBeRemoved, guidelineName, "test");
		//
		
//		System.out.println("testInstances attribute number: " + testInstances.numAttributes());
		Instances attributesRemovedTestInstances = removeCertainAttributes(testInstances,
				attributesToBeRemoved, testGuidelineName, "");
		
//		System.out.println("attributesRemovedTestInstances attribute number: " + attributesRemovedTestInstances.numAttributes());
		// Mark the last attribute in each instance as the true class.
		attributesRemovedTestInstances.setClassIndex(attributesRemovedTestInstances.numAttributes() - 1);

		
		int totalPositiveCases = getPositiveCaseCountInADatasetInstances(attributesRemovedTestInstances);

		// System.out.println("Total positive cases in this guideline: " +
		// totalPositiveCases);

		String[][] probability = generateSortedProbabilityArray(classifier, attributesRemovedTestInstances);

		// System.out.println("Precision at top 10: "
		// + topKPrecisionWODuplicates(probability, 10, guidelineName));
		//
		// System.out.println("Precision at top 20: "
		// + topKPrecisionWODuplicates(probability, 20, guidelineName));

		// this method doesn't exclude the overlapped guideline citation
		// between training and test
		// System.out.println("Precision at top 10: "
		// + topKPrecisionWDuplicates(probability, 10));
		// System.out.println("Precision at top 20: "
		// + topKPrecisionWDuplicates(probability, 20));

//		 int index = 0;
//		 for (final String[] s : probability) {
//		 System.out.printf("%10s %4s %4s %5.4f ", s[0], s[1], s[2],
//		 Double.valueOf(s[3]));
//		 System.out.println("");
//		 if (index++ >= 19) {
//		 break;
//		 }
//		 }


//		// The following code is used to confirm the java API results
//		// are same as the Weka GUI result during hyper-parameter
//		// optimization. Particularly for general evaluations
//		// such as precision, recall and f1
//		Evaluation eval = new Evaluation(testInstances);
//		eval.evaluateModel(model, testInstances);

		
		double[] generalEvaluations = getGeneralEvaluationsOnClassifier(probability, testGuidelineName, true);
		
		if (eval.equals(evaluations[0])) {
			System.out.println(topKPrecisionWDuplicates(probability, 20));
			return topKPrecisionWDuplicates(probability, 20);

		}

		else if (eval.equals(evaluations[1])) {
			System.out.println(topKMeanAveragePrecisionWDuplicates(probability, 20, totalPositiveCases));
			return topKMeanAveragePrecisionWDuplicates(probability, 20, totalPositiveCases);
		}

		else if (eval.equals(evaluations[2])) {
			System.out.println(topKMeanReciprocalRankWDuplicates(probability, 20));
			return topKMeanReciprocalRankWDuplicates(probability, 20);
		}
		

		else if (eval.equals(evaluations[3])) {			
			
//			// The following code is used to confirm the java API results
//			// are same as the Weka GUI result during hyper-parameter
//			// optimization.
//			// But in my application, I use some customized code as the
//			// calculation of recall is different
//			System.out.println("Weka precision 1: " + eval.precision(1));
//			System.out.printf("Weka weighted recall: " + "%4.3f", eval.weightedPrecision());
//			// System.out.printf("%4.3f", eval.weightedPrecision());
//			System.out.println();
//			return eval.precision(1);

			System.out.println(generalEvaluations[0]);
			return generalEvaluations[0];
		}

		else if (eval.equals(evaluations[4])) {

//			// The following code is used to confirm the java API results
//			// are same as the Weka GUI result during hyper-parameter
//			// optimization.
//			// But in my application, I use some customized code as the
//			// calculation of recall is different
//			System.out.println("Weka recall 1: " + eval.recall(1));
//			System.out.printf("Weka weighted recall: " + "%4.3f", eval.weightedRecall());
//			// System.out.printf("%4.3f", eval.weightedRecall());
//			System.out.println();
//			return eval.recall(1);

			System.out.println(generalEvaluations[1]);
			return generalEvaluations[1];

		} else if (eval.equals(evaluations[5])) {


//			// The following code is used to confirm the java API results
//			// are same as the Weka GUI result during hyper-parameter
//			// optimization.
//			// But in my application, I use some customized code as the
//			// calculation of recall is different
//			// System.out.println(eval.fMeasure(1));			
//			System.out.printf("%4.3f", eval.weightedFMeasure());
//			System.out.println();
//			return eval.fMeasure(1);

			System.out.println(generalEvaluations[2]);
			return generalEvaluations[2];
		}

		// if (top_K_eval.equals(evaluations[0])) {
		// return topKPrecisionWODuplicates(probability, 20,
		// testGuidelineName);
		// }
		//
		// else if (top_K_eval.equals(evaluations[1])) {
		// return topKMeanAveragePrecisionWODuplicates(probability, 20,
		// totalPositiveCases, testGuidelineName);
		// }
		//
		// else if (top_K_eval.equals(evaluations[2])) {
		// return topKMeanReciprocalRankWODuplicates(probability, 20,
		// testGuidelineName);
		// }

		// when it returns 0.00000000, something is wrong!
		return 0.00000000;

	}
	
	
	
	public static void testCostMatrix() throws Exception {
		
		// http://www.codemiles.com/weka-examples/cost-sensitive-classifier-random-forest-java-in-weka-t11129.html?mobile=on
		CostSensitiveClassifier costSensitiveClassifier = new CostSensitiveClassifier();
		CostMatrix costMatrix = new CostMatrix(2);
//		CostMatrix costMatrix = new CostMatrix(1);
		
		setChosenHyperParaForEachClassifier();
		
		for (double db = 5.0d; db <= 20.0d; db = db + 5.0d) {
			System.out.println("db: " + db + "\n");
			for (int i = 0; i < classifiers.size(); i++) {
				System.out.println(classifiers.get(i).getClass().toString());

				// costMatrix.setCell(0, 0, 1.0d);
				// costMatrix.setCell(0, 0, db);
				// costMatrix.setCell(1, 1, 2.0d);
				// costMatrix.setCell(1, 1, 1.0d);

				costMatrix.setCell(0, 1, db);
				costMatrix.setCell(1, 0, 2d);

				// costMatrix.setCell(1, 0, db);
				// costMatrix.setCell(0, 1, 2d);

				costSensitiveClassifier.setClassifier(classifiers.get(i));
				costSensitiveClassifier.setCostMatrix(costMatrix);
				
				// //don't need to tune this parameter by Amin
				// costSensitiveClassifier.setMinimizeExpectedCost(true);
				
				for (String s : evaluations) {
					System.out.print(s + "  ");
					trainAClassifierOnADatasetThenTestOnAnotherDataset(costSensitiveClassifier, "hf2014", "MDD", s, "All_Attributes");
				}
				
				System.out.println("---------------");
			}
			System.out.println("==============================\n");
			
//				System.out.println();
		}
		
		// clear the classifiers list for the next possible round
		classifiers.clear();
       
	}

	


	public static void setChosenHyperParaForEachClassifier() throws Exception {		

		// // the following hyper parameters are good for datasets/experiments after the
		// // over-sampling approaches
		
		// IBk ibk = new IBk();
		// ibk.setDistanceWeighting(new SelectedTag(2,
		// weka.classifiers.lazy.IBk.TAGS_WEIGHTING));
		// ibk.setKNN(2);
		// classifiers.add(ibk);
		//
		// NaiveBayes nb = new NaiveBayes();
		// nb.setUseKernelEstimator(false);
		// classifiers.add(nb);
		//
		// BayesNet bn = new BayesNet();
		// classifiers.add(bn);
		//
		// NaiveBayesMultinomial nbm = new NaiveBayesMultinomial();
		// classifiers.add(nbm);
		//
		// Logistic logis = new Logistic();
		// classifiers.add(logis);
		//
		// MultilayerPerceptron mlp = new MultilayerPerceptron();
		// mlp.setLearningRate(0.7);
		// classifiers.add(mlp);
		//
		// SimpleLogistic sl = new SimpleLogistic();
		// classifiers.add(sl);
		//
		// SGD sgd = new SGD();
		// SelectedTag st = new SelectedTag(1,
		// weka.classifiers.functions.SGD.TAGS_SELECTION);
		// sgd.setLossFunction(st);
		// classifiers.add(sgd);
		//
		// DecisionTable dt = new DecisionTable();
		// classifiers.add(dt);
		//
		// J48 j48 = new J48();
		// j48.setReducedErrorPruning(false);
		// j48.setConfidenceFactor((float) 0.5);
		// classifiers.add(j48); 
		
		
//		 // the following hyper parameters are good for datasets/experiments
//		 // with the original size (15845 points) (NO high Impact Journal)
//		
//		IBk ibk = new IBk();
//		ibk.setDistanceWeighting(new SelectedTag(1, weka.classifiers.lazy.IBk.TAGS_WEIGHTING));
//		ibk.setKNN(8);
//		classifiers.add(ibk);		
//		
//		NaiveBayes nb = new NaiveBayes();
//		nb.setUseKernelEstimator(false);
//		classifiers.add(nb);
//
//		BayesNet bn = new BayesNet();
//		classifiers.add(bn);
//
//		NaiveBayesMultinomial nbm = new NaiveBayesMultinomial();
//		classifiers.add(nbm);
//
//		Logistic logis = new Logistic();
//		classifiers.add(logis);
//
//		MultilayerPerceptron mlp = new MultilayerPerceptron();
//		mlp.setLearningRate(0.2);
//		classifiers.add(mlp);
//
//		SimpleLogistic sl = new SimpleLogistic();
//		classifiers.add(sl);
//
//		SGD sgd = new SGD();
//		SelectedTag st = new SelectedTag(1, weka.classifiers.functions.SGD.TAGS_SELECTION);
//		sgd.setLossFunction(st);
//		classifiers.add(sgd);
//
//		DecisionTable dt = new DecisionTable();
//		dt.setSearch(new weka.attributeSelection.GreedyStepwise()); 
//		classifiers.add(dt);
//
//		J48 j48 = new J48();
//		j48.setReducedErrorPruning(false);
//		j48.setConfidenceFactor((float) 0.2);
//		classifiers.add(j48); 				
		
		
//		 // the following hyper parameters are good for datasets/experiments
//		 // with the original size (15845 points) (NO high Impact Journal)
//		
//		IBk ibk = new IBk();
//		ibk.setDistanceWeighting(new SelectedTag(1, weka.classifiers.lazy.IBk.TAGS_WEIGHTING));
//		ibk.setKNN(10);
//		classifiers.add(ibk);		
//		
//		NaiveBayes nb = new NaiveBayes();
//		nb.setUseKernelEstimator(false);
//		classifiers.add(nb);
//
//		BayesNet bn = new BayesNet();
//		classifiers.add(bn);
//
//		NaiveBayesMultinomial nbm = new NaiveBayesMultinomial();
//		classifiers.add(nbm);
//
//		Logistic logis = new Logistic();
//		classifiers.add(logis);
//
//		MultilayerPerceptron mlp = new MultilayerPerceptron();
//		mlp.setLearningRate(0.1);
//		classifiers.add(mlp);
//
//		SimpleLogistic sl = new SimpleLogistic();
//		classifiers.add(sl);
//
//		SGD sgd = new SGD();
//		SelectedTag st = new SelectedTag(1, weka.classifiers.functions.SGD.TAGS_SELECTION);
//		sgd.setLossFunction(st);
//		classifiers.add(sgd);
//
//		DecisionTable dt = new DecisionTable();
//		dt.setSearch(new weka.attributeSelection.GreedyStepwise()); 
//		classifiers.add(dt);
//
//		J48 j48 = new J48();
//		j48.setReducedErrorPruning(false);
//		j48.setConfidenceFactor((float) 0.2);
//		classifiers.add(j48); 
		
		
//		// this part is just for testing the support vector machine combinations		
//		SMO svm1 = new SMO();
//		svm1.setKernel(new weka.classifiers.functions.supportVector.NormalizedPolyKernel());
//		SMO svm2 = new SMO();
//		svm2.setKernel(new weka.classifiers.functions.supportVector.PolyKernel());
//		SMO svm3 = new SMO();
//		svm3.setKernel(new weka.classifiers.functions.supportVector.Puk());
//		SMO svm4 = new SMO();
//		svm4.setKernel(new weka.classifiers.functions.supportVector.RBFKernel());
//		classifiers.add(svm1); 
//		classifiers.add(svm2); 
//		classifiers.add(svm3); 
//		classifiers.add(svm4); 		
		
		
		// this following part is for random forest
		RandomForest rf = new RandomForest();
		rf.setMaxDepth(3);
		rf.setNumTrees(26);
		rf.setNumFeatures(9);
		classifiers.add(rf);
		
		
		for (int i = 0; i < classifiers.size(); i++) {
			System.out.println(classifiers.get(i).getClass().toString());
			for (String s : evaluations) {
				System.out.print(s + "  ");
				trainAClassifierOnADatasetThenTestOnAnotherDataset(classifiers.get(i), "hf2014", "MDD", s, "All_Attributes");
			}
			System.out.println();
		}
				
	}

	
	
	// this method is used to get the HyperPara_Combinations_For_Each_Classifier
	// Criteria =====>
	// check topKPrecision first; if equal, then topKMeanAveragePrecision
	// Result =====>
	// KNN(iBK):
	// tried -(K=1 to 10;All nearest Search algorithms;All distance weighting);
	// Get- KNN=2; distance weighting=Weight by 1/distance; Any Search Algo;
	// Naive Bayes:
	// tried: Use kernel estimator (yes or no)
	// get: Use kernel estimator (no)
	// BayesNet: tried: (All possible estimator; All possible search algorithm)
	// Get: After the NominalToBinary Filter, all the combination produce the
	// same results. Also, result with the filter better than the one without
	// filter)
	// Naive Bayes Multinominal: tried: (no parameter) Get: (no parameter)
	// Logistic: tried: (no parameter) Get: (no parameter)
	// MultiLayerPerceptron
	// Tried:Learning rate from 0.1 to 0.9; Get: rate = 0.7.
	// SimpleLogistic: tried: (no parameter) Get: (no parameter)
	// SGD: tried: All loss functions; Get: loss function = Log loss (logistic
	// regression);
	// DecisionTable: tried: All search; get: search doesn't matter, use any;
	// J48: tried: (reduced error pruning (yes or no); confidence factor (0.1 to
	// 0.75)) Get: (reduced error pruning (no) + confidence factor 0.5)
	public static void tryHyperParaCombinationsForEachClassifier() throws Exception {		
//		// part of source code for the following
//		// http://grepcode.com/file/repo1.maven.org/maven2/nz.ac.waikato.cms.weka/weka-dev/3.7.12/weka/classifiers/lazy/IBk.java#IBk.setDistanceWeighting%28weka.core.SelectedTag%29
//		IBk ibk = new IBk();
//		final int WEIGHT_NONE = 1;	// No distance weighting	
//		final int WEIGHT_INVERSE = 2;// weight by 1/distance. 		
//		final int WEIGHT_SIMILARITY = 4; // weight by 1-distance.
////		final Tag[] TAGS_WEIGHTING = { new Tag(WEIGHT_NONE, "No distance weighting"),
////				new Tag(WEIGHT_INVERSE, "Weight by 1/distance"), new Tag(WEIGHT_SIMILARITY, "Weight by 1-distance") };
//		int[] weights = { WEIGHT_NONE, WEIGHT_INVERSE, WEIGHT_SIMILARITY };
//		NearestNeighbourSearch[] nbs = { 
//				new weka.core.neighboursearch.BallTree(),
////				new weka.core.neighboursearch.CoverTree(),
////				new weka.core.neighboursearch.FilteredNeighbourSearch(),
////				new weka.core.neighboursearch.KDTree(),
////				new weka.core.neighboursearch.LinearNNSearch()
//		};
//		for (int knn = 1; knn <= 10; knn++) {
//			for (int dw = 0; dw < weights.length; dw++) {
//				for (int nnsa = 0; nnsa < nbs.length; nnsa++) {
////					http://stackoverflow.com/questions/26701280/how-to-set-evaluation-criteria-of-weka-grid-search-through-java-code
//					SelectedTag st=new SelectedTag(weights[dw], weka.classifiers.lazy.IBk.TAGS_WEIGHTING);
//					ibk.setDistanceWeighting(st);
//					ibk.setKNN(knn);
//					ibk.setNearestNeighbourSearchAlgorithm(nbs[nnsa]);
//					System.out.println(
//							"KNN=" + ibk.getKNN()
//							+"; distance weighting="+ ibk.getDistanceWeighting().getSelectedTag().getReadable().toString()
//							+ "; Search algorithms=" + ibk.getNearestNeighbourSearchAlgorithm().toString());
//					for (String s : evaluations) {
//						System.out.print(s + "  ");
//						trainAClassifierOnADatasetThenTestOnAnotherDataset(ibk, "hf2014", "MDD", s, "All_Attributes");
//					}
//					System.out.println();
//				}
//			}
//		}
//		System.out.println();
//
//		
//		
//		NaiveBayes nb = new NaiveBayes();
//		boolean[] useKernelEstimator = { true, false };
//		for (int i = 0; i < useKernelEstimator.length; i++) {
//			nb.setUseKernelEstimator(useKernelEstimator[i]);
//			System.out.println(nb.getClass().toString() + " " + nb.getUseKernelEstimator());
//			classifiers.add(nb);
//			for (String s : evaluations) {
//				System.out.print(s + " ");
//				trainAClassifierOnADatasetThenTestOnAnotherDataset(nb, "hf2014", "MDD", s, "All_Attributes");
//			}
//		}
//		System.out.println();
//
//		
//		
//		 BayesNet bn = new BayesNet();
//		 BayesNetEstimator[] bnEstimator = { 
//			 new weka.classifiers.bayes.net.estimate.SimpleEstimator(), 
//			 new weka.classifiers.bayes.net.estimate.BMAEstimator(),
////		 new weka.classifiers.bayes.net.estimate.BayesNetEstimator(),
////		 new weka.classifiers.bayes.net.estimate.MultiNomialBMAEstimator(),
//		 };		
//		 SearchAlgorithm[] SearchAlgorithm = { 
////		 new weka.classifiers.bayes.net.search.local.GeneticSearch(),
//		 new weka.classifiers.bayes.net.search.local.HillClimber(),
//		 new weka.classifiers.bayes.net.search.local.K2(),
////		 new weka.classifiers.bayes.net.search.local.LAGDHillClimber(),
//		 new weka.classifiers.bayes.net.search.local.RepeatedHillClimber(),
////		 new weka.classifiers.bayes.net.search.local.SimulatedAnnealing(),
//		 new weka.classifiers.bayes.net.search.local.TabuSearch(),
////		 new weka.classifiers.bayes.net.search.local.TAN() 
//		 };
//		for (int bnEs = 0; bnEs < bnEstimator.length; bnEs++) {
//			for (int bnsa = 0; bnsa < SearchAlgorithm.length; bnsa++) {
//				bn.setEstimator(bnEstimator[bnEs]);
//				bn.setSearchAlgorithm(SearchAlgorithm[bnsa]);
//				classifiers.add(bn);
//				System.out.println(bn.getClass().toString() + " " + bn.getEstimator() + " "
//						+ bn.getSearchAlgorithm().getClass().toString());
//				for (String s : evaluations) {
//					System.out.print(s + " ");
//					trainAClassifierOnADatasetThenTestOnAnotherDataset(bn, "hf2014", "MDD", s, "All_Attributes");
//				}
//			}
//		}
//		System.out.println();
//		
//		
//
//		NaiveBayesMultinomial nbm = new NaiveBayesMultinomial();
//		System.out.println(nbm.getClass().toString());
//		for (String s : evaluations) {
//			System.out.print(s + " ");
//			trainAClassifierOnADatasetThenTestOnAnotherDataset(nbm, "hf2014", "MDD", s, "All_Attributes");
//		}
//		System.out.println();
//		
//		
//
//		 Logistic logis = new Logistic();
//		 System.out.println(logis.getClass().toString());
//		 classifiers.add(logis);
//		 for (String s : evaluations) {
//		 System.out.print(s + " ");
//		 trainAClassifierOnADatasetThenTestOnAnotherDataset(logis,
//		 "hf2014", "MDD", s, "All_Attributes");
//		 }		
//		 System.out.println();
//
//		
//		
//		 MultilayerPerceptron mlp = new MultilayerPerceptron();		
//		 for (double lr = 0.1; lr <= 0.9; lr = lr + 0.1) {
//		 mlp.setLearningRate(lr);
//		 classifiers.add(mlp);
//		 System.out.println(mlp.getClass().toString() + " " +
//		 mlp.getLearningRate());		
//		 for (String s : evaluations) {
//		 System.out.print(s + " ");
//		 trainAClassifierOnADatasetThenTestOnAnotherDataset(mlp,
//		 "hf2014", "MDD", s, "All_Attributes");
//		 }
//		 System.out.println();
//		 }
//		 
//
//		
//		SimpleLogistic sl = new SimpleLogistic();
//		System.out.println(sl.getClass().toString());
//		classifiers.add(sl);
//		for (String s : evaluations) {
//			System.out.print(s + " ");
//			trainAClassifierOnADatasetThenTestOnAnotherDataset(sl, "hf2014", "MDD", s, "All_Attributes");
//		}
//		System.out.println();
//
//		
//		
//		// Here is the source for the following part of code
//		// http://grepcode.com/file/repo1.maven.org/maven2/nz.ac.waikato.cms.weka/weka-dev/3.7.12/weka/classifiers/functions/SGD.java#SGD
//		// final int HINGE = 0;// the hinge loss function.
//		// final int LOGLOSS = 1; // the log loss function.
//		// final int SQUAREDLOSS = 2;// the squared loss function.
//		// final int EPSILON_INSENSITIVE = 3;//The epsilon insensitive loss
//		// function
//		// final int HUBER = 4; // The Huber loss function
//		SGD sgd = new SGD();		
//		// I only tested HINGE and LOGLOSS because the other three functions are
//		// firing errors
//		for (int i = 0; i < 2; i++) {			
//			// Struggled for a while here, then I got the idea from:
//			// http://stackoverflow.com/questions/26701280/how-to-set-evaluation-criteria-of-weka-grid-search-through-java-code
//			SelectedTag st = new SelectedTag(i, weka.classifiers.functions.SGD.TAGS_SELECTION);
//			sgd.setLossFunction(st);		
//			System.out.println(
//					sgd.getClass().toString() + " " + sgd.getLossFunction().getSelectedTag().getReadable().toString());
//			for (String s : evaluations) {
//				System.out.print(s + " ");
//				trainAClassifierOnADatasetThenTestOnAnotherDataset(sgd, "hf2014", "MDD", s, "All_Attributes");
//			}			
//		}		
//		System.out.println();
//		
//		
//	
//		DecisionTable dt = new DecisionTable();
//		ASSearch[] ats = { new weka.attributeSelection.BestFirst(), new weka.attributeSelection.GreedyStepwise(),
////				new weka.attributeSelection.Ranker()
//		};
//		for (int atss = 0; atss < ats.length; atss++) {
//			dt.setSearch(ats[atss]);
//			System.out.println(dt.getClass().toString() + " " + dt.getSearch());
//			for (String s : evaluations) {
//				System.out.print(s + " ");
//				trainAClassifierOnADatasetThenTestOnAnotherDataset(dt, "hf2014", "MDD", s, "All_Attributes");
//			}
//		}
//		System.out.println();
//		
		
		
//		J48 j48 = new J48();
//		boolean[] reduceErrorPruning = { true, false };
//		double[] confidenceFactor = { 0.1, 0.2, 0.3, 0.4, 0.5,
//				// 0.6, 0.7, 0.75
//		};		
//		for (int ep = 0; ep < reduceErrorPruning.length; ep++) {
//			for (int cfF = 0; cfF < confidenceFactor.length; cfF++) {
//				j48.setReducedErrorPruning(reduceErrorPruning[ep]);
//				j48.setConfidenceFactor((float) confidenceFactor[cfF]);
//				// classifiers.add(j48);
//				System.out.println(j48.getClass().toString() + " " + j48.getReducedErrorPruning() + " "
//						+ j48.getConfidenceFactor());
//				for (String s : evaluations) {
//					System.out.print(s + " ");
//					trainAClassifierOnADatasetThenTestOnAnotherDataset(j48, "hf2014", "MDD", s, "All_Attributes");
//				}
//			}
//		}
		
//		SMO svm = new SMO();
//		Kernel[] kernelValues = new Kernel[] { 
//				new weka.classifiers.functions.supportVector.NormalizedPolyKernel(),
//				new weka.classifiers.functions.supportVector.PolyKernel(),
////				new weka.classifiers.functions.supportVector.PrecomputedKernelMatrixKernel(),
//				new weka.classifiers.functions.supportVector.Puk(),
//				new weka.classifiers.functions.supportVector.RBFKernel(),
////				new weka.classifiers.functions.supportVector.StringKernel() 
//				};
//		for (int i = 0; i < kernelValues.length; i++) {
//			svm.setKernel(kernelValues[i]);
//			System.out.println(svm.getClass().toString() + " " + svm.getKernel());
//			for (String s : evaluations) {
//				System.out.print(s + " ");
//				trainAClassifierOnADatasetThenTestOnAnotherDataset(svm, "hf2014", "MDD", s, "All_Attributes");
//			}
//		}
		
		RandomForest rf = new RandomForest();
		for (int maxDepth = 1; maxDepth <= 19; maxDepth = maxDepth + 2) {
			for (int numTress = 1; numTress <= 51; numTress = numTress + 5) {
				for (int numFeatures = 2; numFeatures <= 13; numFeatures++) {
					rf.setMaxDepth(maxDepth);
					rf.setNumTrees(numTress);
					rf.setNumFeatures(numFeatures);
					System.out.println(rf.getClass().toString() + " " + " MaxDepth: " + rf.getMaxDepth() + " NumTrees: "
							+ rf.getNumTrees() + " NumFeatures: " + rf.getNumFeatures());

					for (String s : evaluations) {
						System.out.print(s + " ");
						trainAClassifierOnADatasetThenTestOnAnotherDataset(rf, "hf2014", "MDD", s,
								"All_Attributes");
					}

				}
			}
		}
		
	}

	// modified from method test_All_Combinations()
	// refer to it for all the previous tries
	public static void testHypothesesUsingTheTunedClassifier() {

		for (String attrChoice : attributes) {
//			System.out.println("Attributes: ");
			System.out.println(attrChoice);
			System.out.println("======================================");

			for (String s : evaluations) {

				System.out.println("Evaluation: " + s);

				double[] localPrecisionsArray = new double[guidelines.length];

				for (int i = 0; i < guidelines.length; i++) {

					System.out.print(guidelines[i] +": ");

					try {
						
						
						RandomForest rf = new RandomForest();
						rf.setMaxDepth(3);
						rf.setNumTrees(26);
						rf.setNumFeatures(9);
						localPrecisionsArray[i] = trainAClassifierOnADatasetThenTestOnAnotherDataset(
								rf, "hf2014", guidelines[i], s, attrChoice);
						
//						// Without over-sampling approaches, the final classifier we identified is:
//						// NaiveBayes() without hyper parameter or cost matrix
//						localPrecisionsArray[i] = trainAClassifierOnADatasetThenTestOnAnotherDataset(
//								new NaiveBayes(), "hf2014", guidelines[i], s, attrChoice);
						
//						// Without over-sampling approaches using the data set
//						// 15845 data points (no high impact journal feature), 
//						// we found that the final classifier we identified is :
//						// logistic (default hyper parameters) + cost Matrix (20 to 2)
//						CostSensitiveClassifier costSensitiveClassifier = new CostSensitiveClassifier();
//						CostMatrix costMatrix = new CostMatrix(2);
//						costMatrix.setCell(0, 1, 20.0d);
//						costMatrix.setCell(1, 0, 2d);					
//						costSensitiveClassifier.setClassifier(new Logistic());
//						costSensitiveClassifier.setCostMatrix(costMatrix);						
//						localPrecisionsArray[i] = trainAClassifierOnADatasetThenTestOnAnotherDataset(
//								costSensitiveClassifier, "hf2014", guidelines[i], s, attrChoice);

						////as a negative control to test whether cost matrix related codes work 
						//trainAClassifierOnADatasetThenTestOnAnotherDataset(new Logistic(), "hf2014", "MDD", s, "All_Attributes");
						
//						// With oversampling approaches, the final classifier we identified is:
//						// BayesNet without hyper parameter or cost matrix
//						localPrecisionsArray[i] = trainAClassifierOnADatasetThenTestOnAnotherDataset(
//								new BayesNet(), "hf2014", guidelines[i], s, attrChoice);
						//// localPrecisionsArray[i] =
						//// trainAClassifierOnADatasetThenTestOnAnotherDataset(new
						//// NaiveBayes(), "hf2014", guidelines[i], s,
						//// attrChoice);
						//// localPrecisionsArray[i] =
						//// trainAClassifierOnADatasetThenTestOnAnotherDataset(new
						//// NaiveBayesMultinomial(), "hf2014", guidelines[i], s,
						//// attrChoice);

//						System.out.println("From Array: " + localPrecisionsArray[i]);
//						System.out.println();
						
					} catch (Exception e) {
						e.printStackTrace();
					}
				}

				Statistics sta = new Statistics(localPrecisionsArray);
				// // System.out.println("Mean: " + sta.getMean());
				//// System.out.println("Top 10 precision statistics: ");
				//// System.out.print("Mean: ");
				System.out.printf("%3.2f", sta.getMean());
				System.out.print("±");
				// System.out.println(" ");

				//
				// System.out.print("SD: ");
				System.out.printf("%3.2f", sta.getStdDev());
				// System.out.print(" ");
				System.out.println("\n");
				//
				//// System.out.println("Median: " + sta.median() + "\n\n");
				// System.out.println(sta.median());

				// resetWeightedGuidelineArray(specificArray);
			}
			System.out.println();
//			System.out.println();
			// System.out.println("======================================");
		}

	}
	
	
	public static void testAllCombinations() {

		for (String attrChoice : attributes) {
			System.out.println("Attributes: ");
			System.out.println(attrChoice);
			System.out.println("======================================");
			for (Classifier c : classifiers) {
				// System.out.println(classifiers.indexOf(c) + 1);

				System.out.println(c.getClass().getTypeName() + " ==>");

				// System.out.print(c.getClass().getTypeName() + " ==> ");

				for (String s : evaluations) {

					System.out.println("Evaluation: " + s);

					double[] localPrecisionsArray = new double[guidelines.length];

					Date start = new Date();
					// System.out.println();

					for (int i = 0; i < guidelines.length; i++) {

						System.out.println(guidelines[i]);

						// this part is for comparing the performance of
						// models as opposed to classifier
						// per Gang's suggestion
						try {

							// localPrecisionsArray[i] =
							// getModelForClassifierThenTestOnDataset(c,
							// attrChoice,
							// guidelines[i], s, "6 + 5/");

							localPrecisionsArray[i] = getModelForClassifierThenTestOnDataset(c, attrChoice,
									guidelines[i], s, "1 + 10/");

							System.out.println(getModelForClassifierThenTestOnDataset(c, attrChoice,
									guidelines[i], s, "1 + 10/"));
						} catch (Exception e) {
							e.printStackTrace();
						}

						// try {
						// trainAClassifier(c, attrChoice, guidelines[i]);
						// // } catch (Exception e) {
						// // e.printStackTrace();
						// // }
						// //
						// // try {
						//
						// // updateWeightedGuidelineArray(specificArray,
						// // testAClassifier(c,attrChoice, guidelines[i],
						// // s), guidelines[i]);
						//
						// // precisionsArray[i] = testAClassifier(c,
						// // attrChoice, guidelines[i], s);
						//
						// localPrecisionsArray[i] = testAClassifier(c,
						// attrChoice, guidelines[i], s);
						// } catch (Exception e) {
						// e.printStackTrace();
						// }

						// System.out.println();
					}

					Date end = new Date();

					// System.out.println("Duration: "+
					// (end.getTime()-start.getTime()) + " milliseconds");
					System.out.print((end.getTime() - start.getTime()) / 1000 + " ");

					// System.out.println("Updated Weighted Array: ");
					// System.out.println(Arrays.deepToString(specificArray));

					// System.out.println(s + " Weighed Mean: "
					// + getWeightedGuidelineMeanValue(specificArray)
					// + "\n\n");

					// System.out.println(getWeightedGuidelineMeanValue(specificArray)
					// + "\n");

					// System.out.println("Evaluation: " + s);

					Statistics sta = new Statistics(localPrecisionsArray);
					// // System.out.println("Mean: " + sta.getMean());
					//// System.out.println("Top 10 precision statistics: ");
					//// System.out.print("Mean: ");
					System.out.printf("%3.2f", sta.getMean());
					System.out.print("±");
					// System.out.println(" ");

					//
					// System.out.print("SD: ");
					System.out.printf("%3.2f", sta.getStdDev());
					// System.out.print(" ");
					System.out.println("\n");
					//
					//// System.out.println("Median: " + sta.median() + "\n\n");
					// System.out.println(sta.median());

					// resetWeightedGuidelineArray(specificArray);
				}
				System.out.println();
			}

			System.out.println();
			// System.out.println("======================================");
		}

	}

	// This method is used to generate the SMOTE dataset
	public static void generateSMOTEDatasets() throws Exception {

		DataSource source;

		String filesFolder = filesRootPath + "Auto_Weka_DataSets/";

		File dir = new File(filesFolder);
		File[] directoryListing = dir.listFiles();
		if (directoryListing != null) {
			for (File child : directoryListing) {

				source = new DataSource(child.toString());
				// System.out.println(child.getName());

				Instances data = source.getDataSet();
				if (data.classIndex() == -1) {
					data.setClassIndex(data.numAttributes() - 1);
				}

				for (int i = 100; i <= 900; i = i + 100) {
					// Got this SMOTE jar from the below, and add it into
					// Referenced Library:
					// http://www.java2s.com/Code/Jar/s/Downloadsmote103jar.htm
					SMOTE filters = new SMOTE();
					filters.setInputFormat(data);

					filters.setPercentage(i);
					Instances subSamplingInstances = Filter.useFilter(data, filters);
					// System.out.println(i);
					int fold = i / 100 + 1;

					// The regular expression used here, the reference is
					// http://stackoverflow.com/questions/924394/how-to-get-the-filename-without-the-extension-in-java

					String filepath = filesRootPath + "Auto_Weka_DataSets/SMOTE/"
							+ child.getName().replaceFirst("[.][^.]+$", "") + "_X" + fold + ".arff";

					System.out.println(filepath);

					// Thread.sleep(10000000);

					ArffSaver saver = new ArffSaver();
					saver.setInstances(subSamplingInstances);

					saver.setFile(new File(filepath));
					saver.writeBatch();
				}

				// write the instance to certain folder;

				// return null;

				// return subSamplingInstances;

				// Do something with child
			}
		} else {
			System.out.println("Something is wrong with this folder, Please double check!");
		}
	}

	// to update the array value for the weighted guideline
	public static void resetWeightedGuidelineArray(String[][] wga) {

		for (int i = 0; i < wga.length; i++) {
			wga[i][2] = "0.000";
		}

	}

	// to update the array value for the weighted guideline
	public static void updateWeightedGuidelineArray(String[][] wga, double eval, String guidelineName) {

		for (int i = 0; i < wga.length; i++) {
			if (wga[i][0].equals(guidelineName)) {
				wga[i][2] = Double.toString(eval);
			}
		}

	}

	// getWeightedGuidelineMeanValue for a 2-D array;
	public static double getWeightedGuidelineMeanValue(String[][] wga) {

		double tmp = 0.000;

		for (int i = 0; i < wga.length; i++) {
			tmp = tmp + Double.parseDouble(wga[i][1]) * Double.parseDouble(wga[i][2]);
		}
		return tmp;
	}

	public static Instances getTrainingOrTestInstancesForAGuideline(String guidelineName,
			String trainingOrTest) throws Exception {
		DataSource source;
		String filename = filesRootPath + specificFolderPath + guidelineName + "/" + trainingOrTest + ".arff";
		source = new DataSource(filename);
		Instances data = source.getDataSet();
		if (data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes() - 1);
		}
		return data;
	}

	// https://weka.wikispaces.com/file/view/CrossValidationAddPrediction.java
	public static void runTenFoldOnAGuideline(String dataset) throws Exception {

		// String filename = "";
		// if (dataset.equals("Original")) {
		// filename = filesRootPath + specificFolderPath +
		// "six_Guidelines_Deduplicated.arff";
		// }
		// else if (dataset.equals("Attributes_Deduplicated")) {
		// filename = filesRootPath + specificFolderPath
		// + "six_Guidelines_Deduplicated.arff";
		// }

		// this part is for testing the Auto_weka hyperparameter optimization
		String filename = "E:/Studying/autoweka-0.5/datasets/MDD_Deduplicated_SMOTE_X_5_K1.arff";

		// loads data and set class index
		DataSource source = new DataSource(filename);
		Instances data = source.getDataSet();

		System.out.println();
		System.out.println("Positive_Case_Count: " + getPositiveCaseCountInADatasetInstances(data));

		data.setClassIndex(data.numAttributes() - 1);

		for (Classifier cls : classifiers) {
			System.out.println();
			Date start = new Date();
			System.out.println(cls.getClass().getTypeName() + " ==>");
			try {

				// classifier
				// Classifier cls = classifiers[0];

				// other options
				int seed = 0;
				int folds = 10;

				double[] topKPrecisionsArray = new double[10];


				// randomize data
				Random rand = new Random(seed);
				Instances randData = new Instances(data);
				randData.randomize(rand);

				if (randData.classAttribute().isNominal())
					randData.stratify(folds);

				// perform cross-validation and add predictions
				Instances predictedData = null;
				Evaluation eval = new Evaluation(randData);
				for (int n = 0; n < folds; n++) {
					Instances train = randData.trainCV(folds, n);
					Instances test = randData.testCV(folds, n);
					// the above code is used by the StratifiedRemoveFolds
					// filter, the code below by the Explorer/Experimenter:
					// Instances train = randData.trainCV(folds, n, rand);

					// build and evaluate classifier
					// Classifier clsCopy = Classifier.makeCopy(cls);
					cls.buildClassifier(train);
					eval.evaluateModel(cls, test);

					// eval.crossValidateModel(cls, test, 10, rand, null);
					// System.out.println("Round: " + (n + 1));

		
					// System.out.println("Total Cases:" + test.numInstances());

					String[][] sortedProbabilityArray = generateSortedProbabilityArray(cls, test);

					// this method is used to print a 2-D array
					// System.out.println(Arrays.deepToString(sortedProbabilityArray));

					// System.out.println("Top 20 Precision_W_Duplicates:"
					// + topKPrecisionWDuplicates(sortedProbabilityArray,
					// 20));

					// System.out.println();

					topKPrecisionsArray[n] = topKPrecisionWDuplicates(sortedProbabilityArray, 20);

					// top_K_MeanAveragePrecision_array[n] =
					// topKMeanAveragePrecisionWDuplicates(
					// sortedProbabilityArray, 20, positive_counts);
					//
					// top_K_MeanReciprocalRank_array[n] =
					// topKMeanReciprocalRankWDuplicates(
					// sortedProbabilityArray, 20);

					// add predictions
					AddClassification filter = new AddClassification();
					filter.setClassifier(cls);
					filter.setOutputClassification(true);
					filter.setOutputDistribution(true);
					filter.setOutputErrorFlag(true);
					filter.setInputFormat(train);
					Filter.useFilter(train, filter); // trains the classifier

					// perform predictions on test set
					Instances pred = Filter.useFilter(test, filter);
					if (predictedData == null)
						predictedData = new Instances(pred, 0);
					for (int j = 0; j < pred.numInstances(); j++)
						predictedData.add(pred.instance(j));
				}

				Date end = new Date();
				// System.out.println("Duration: "+
				// (end.getTime()-start.getTime())
				// + " milliseconds");
				System.out.print((end.getTime() - start.getTime()) / 1000 + " ");

				// System.out.println("===========Overall===========");

				// Statistics topKPrecisionsSta = new Statistics(
				// topKPrecisionsArray);
				//
				// // System.out.println("Top K Precision Statistics: ");
				// // System.out.println("Top 10 precision statistics: ");
				// // System.out.print("Mean: ");
				// System.out.printf("%3.2f", topKPrecisionsSta.getMean());
				// System.out.print(" ");
				// System.out.println();

				// // System.out.print("SD: ");
				// System.out.printf("%3.2f", topKPrecisionsSta.getStdDev());
				// System.out.print(" ");
				//
				// // System.out.println("Median: " + sta.median() + "\n\n");
				// System.out.println(topKPrecisionsSta.median());

				// Statistics topKMeanAveragePrecisionSta
				// = new Statistics(top_K_MeanAveragePrecision_array);
				// System.out.println("topKMeanAveragePrecision Statistics:
				// ");
				// // System.out.println("Mean: " + sta.getMean());
				// // System.out.println("Top 10 precision statistics: ");
				// // System.out.print("Mean: ");
				// System.out.printf("%3.2f",
				// topKMeanAveragePrecisionSta.getMean());
				// System.out.print(" ");
				//
				// // System.out.print("SD: ");
				// System.out.printf("%3.2f",
				// topKMeanAveragePrecisionSta.getStdDev());
				// System.out.print(" ");
				//
				// // System.out.println("Median: " + sta.median() + "\n\n");
				// System.out.println(topKMeanAveragePrecisionSta.median());

				// Statistics topKMeanReciprocalRankSta = new
				// Statistics(top_K_MeanReciprocalRank_array);
				// System.out.println("topKMeanReciprocalRank Statistics: ");
				// // System.out.println("Mean: " + sta.getMean());
				// // System.out.println("Top 10 precision statistics: ");
				// // System.out.print("Mean: ");
				// System.out.printf("%3.2f",
				// topKMeanReciprocalRankSta.getMean());
				// System.out.print(" ");
				//
				// // System.out.print("SD: ");
				// System.out.printf("%3.2f",
				// topKMeanReciprocalRankSta.getStdDev());
				// System.out.print(" ");
				//
				// // System.out.println("Median: " + sta.median() + "\n\n");
				// System.out.println(topKMeanReciprocalRankSta.median());

				// output evaluation
				System.out.println();
				System.out.println("=== Setup ===");
				// System.out.println("Classifier: " + cls.getClass().getName()
				// + " " + Utils.joinOptions(cls.getOptions()));
				System.out.println("Dataset: " + data.relationName());
				System.out.println("Folds: " + folds);
				System.out.println("Seed: " + seed);
				System.out.println();
				System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));

				// Weighted Average Measurement: precision, recall, F-measure
				// and ROC Area
				System.out.println("(weighted Precision = " + eval.weightedPrecision() + ")");

				System.out.println("(weighted Recall = " + eval.weightedRecall() + ")");

				System.out.println("(weighted F-Measure = " + eval.weightedFMeasure() + ")");

				System.out.println("(weighted AreaUnderROC = " + eval.weightedAreaUnderROC() + ")");

				System.out.println();

				// Guideline citations Measurement: precision, recall,
				// F-measure and ROC
				System.out.println("(Guideline citations Precision = " + eval.precision(1) + ")");

				System.out.println("(Guideline citations Recall = " + eval.recall(1) + ")");

				System.out.println("(Guideline citations F-Measure = " + eval.fMeasure(1) + ")");

				System.out.println("(Guideline citations AreaUnderROC = " + eval.areaUnderROC(1) + ")");

				System.out.println();

			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public static void runAClassifierOnAGuidelineToGetOverallResults(Classifier model, String guidelineName) {

		try {
			// training a model:
			Instances training = getTrainingOrTestInstancesForAGuideline(guidelineName, "training");
			model.buildClassifier(training);

			// evaluate a model:
			Instances test = getTrainingOrTestInstancesForAGuideline(guidelineName, "test");

			Evaluation eval = new Evaluation(test);

			eval.evaluateModel(model, test);

			System.out.println(eval.toSummaryString("\nResults\n======\n", false));

			System.out.println(eval.toClassDetailsString("\ntoClassDetailsString\n======\n"));

			// System.out.println(eval.precision(2));
			System.out.println(eval.confusionMatrix());

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	// background about this part:
	// https://weka.wikispaces.com/Serialization
	public static double getModelForClassifierThenTestOnDataset(Classifier model,
			String attributesToBeRemoved, String guidelineName, String eval, String choice) throws Exception {

		// get the training data set, on which train the classifier
		// Instances originalTraining =
		// getTrainingOrTestInstancesForAGuideline(
		// guidelineName, "training");

		DataSource source;
		String filename = "";
		if (choice.equals("6 + 5/")) {
			filename = filesRootPath + choice + "six_Guidelines_Deduplicated" + ".arff";
		} else if (choice.equals("1 + 10/")) {
			filename = filesRootPath + choice + "MDD_Deduplicated" + ".arff";
		}

		source = new DataSource(filename);
		Instances training = source.getDataSet();

		if (training.classIndex() == -1) {
			training.setClassIndex(training.numAttributes() - 1);
		}

		// Instances attributesDeduplicatedTraining =
		// removeDuplicatedAttributes(
		// originalTraining, guidelineName, "training");

		// Instances attributesRemovedTraining = removeCertainAttributes(
		// originalTraining, attributesToBeRemoved,
		// guidelineName, "training");
		//
		// attributesRemovedTraining.setClassIndex(attributesRemovedTraining
		// .numAttributes() - 1);

		model.buildClassifier(training);

		// the path the model will be saved into
		String modelStoredPath = filesRootPath + choice + model.getClass().getName();

		// serialize model
		weka.core.SerializationHelper.write(modelStoredPath, model);

		// Deserialize the classifier.
		Classifier classifier = (Classifier) weka.core.SerializationHelper.read(modelStoredPath);

		// Load the test instances.
		Instances originalTestInstances = getTrainingOrTestInstancesForAGuideline(guidelineName, "test");

		Instances attributesDeduplicatedTestInstances = removeDuplicatedAttributes(originalTestInstances,
				guidelineName, "test");

		Instances attributesRemovedTestInstances = removeCertainAttributes(attributesDeduplicatedTestInstances,
				attributesToBeRemoved, guidelineName, "test");

		// Mark the last attribute in each instance as the true class.
		attributesRemovedTestInstances.setClassIndex(attributesRemovedTestInstances.numAttributes() - 1);

		int totalPositiveCases = getPositiveCaseCountInADatasetInstances(attributesRemovedTestInstances);

		// System.out.println("Total positive cases in this guideline: "
		// + totalPositiveCases);

		String[][] probability = generateSortedProbabilityArray(classifier, attributesRemovedTestInstances);

		// System.out.println("Precision at top 10: "
		// + topKPrecisionWODuplicates(probability, 10, guidelineName));
		//
		// System.out.println("Precision at top 20: "
		// + topKPrecisionWODuplicates(probability, 20, guidelineName));

		// this method doesn't exclude the overlapped guideline citation
		// between training and test
		// System.out.println("Precision at top 10: "
		// + topKPrecisionWDuplicates(probability, 10));
		// System.out.println("Precision at top 20: "
		// + topKPrecisionWDuplicates(probability, 20));

		// int index = 0;
		// for (final String[] s : probability) {
		// System.out.printf("%10s %4s %4s %5.4f ", s[0], s[1], s[2],
		// Double.valueOf(s[3]));
		// System.out.println("");
		// if (index++ >= 19) {
		// break;
		// }
		// }

		if (eval.equals(evaluations[0])) {
			return topKPrecisionWODuplicates(probability, 20, guidelineName);
		}

		else if (eval.equals(evaluations[1])) {
			return topKMeanAveragePrecisionWODuplicates(probability, 20, totalPositiveCases, guidelineName);
		}

		else if (eval.equals(evaluations[2])) {
			return topKMeanReciprocalRankWODuplicates(probability, 20, guidelineName);
		}

		// when it returns 0.00000000, something is wrong!
		return 0.00000000;

	}

	// background about this part:
	// https://weka.wikispaces.com/Serialization
	public static void trainAClassifier(Classifier model, String attributesToBeRemoved, String guidelineName)
			throws Exception {

		// reference for the following codes:
		// http://stackoverflow.com/questions/20014412/weka-only-changing-numeric-to-nominal

		// get the training data set, on which train the classifier
		Instances originalTraining = getTrainingOrTestInstancesForAGuideline(guidelineName, "training");

		Instances attributesDeduplicatedTraining = removeDuplicatedAttributes(originalTraining, guidelineName,
				"training");

		Instances attributesRemovedTraining = removeCertainAttributes(attributesDeduplicatedTraining,
				attributesToBeRemoved, guidelineName, "training");

		// Attribute att = attributes_removed_training_0
		// .attribute(attributes_removed_training_0.numAttributes() - 1);
		// attributes_removed_training_0.renameAttributeValue(att, "Yes", "" +
		// 1);
		// attributes_removed_training_0.renameAttributeValue(att, "No", "" +
		// 0);

		// convert nominal attributes to binary
		// Instances attributesRemovedTraining = new Instances(
		// attributes_removed_training_0);

		// // the purpose of setClassIndex to the second last is because I can't
		// set
		// // the last attribute (guideline) from nominal to binary if guideline
		// attribute
		// // is the class, so I set second to last as class first then set it
		// back later
		// attributesRemovedTraining.setClassIndex(attributesRemovedTraining
		// .numAttributes() - 2);
		// NominalToBinary nominalToBinary = new NominalToBinary();
		// nominalToBinary.setInputFormat(attributesRemovedTraining);
		// attributesRemovedTraining = Filter.useFilter(
		// attributesRemovedTraining, nominalToBinary);
		//
		// saveInstancesToAnARFF(attributesRemovedTraining,
		// "De_Duplicated_Binary", guidelineName,
		// "training");

		attributesRemovedTraining.setClassIndex(attributesRemovedTraining.numAttributes() - 1);

		model.buildClassifier(attributesRemovedTraining);

		// // This method will be used to get the specific information about the
		// // NB model. Overall, this part is not working yet, I will come back
		// // to work on after I got all the data ready

		// NaiveBayes nb = (NaiveBayes)model;
		//
		// nb.setDisplayModelInOldFormat(true);
		// System.out.println("nb.toString()");
		// System.out.println(nb.toString());
		//
		// nb.setDisplayModelInOldFormat(false);
		// System.out.println("nb.toString()");
		// System.out.println(nb.toString());
		//
		// System.out.println("nb.getTechnicalInformation()");
		// System.out.println(nb.getTechnicalInformation());
		//
		// Estimator ee = nb.getClassEstimator();
		// System.out.println("ee.toString()");
		// System.out.println(ee.toString());
		//
		// System.out.println("nb.globalInfo()");
		// System.out.println(nb.globalInfo());
		//
		// System.out.println("nb.numDecimalPlacesTipText()");
		// System.out.println(nb.numDecimalPlacesTipText());

		// // the following lines is to test discretizing the data set
		// Discretize filter;
		// // setup filter
		// filter = new Discretize();
		// filter.setInputFormat(attributesRemovedTraining);
		//
		// Instances outputTrain;
		// // apply filter
		// outputTrain = Filter.useFilter(attributesRemovedTraining, filter);
		//// outputTest = Filter.useFilter(inputTest, filter);
		// model.buildClassifier(outputTrain);

		// the path the model will be saved into
		String modelStoredPath = filesRootPath + specificFolderPath + guidelineName + "/"
				+ model.getClass().getName();

		// serialize model
		weka.core.SerializationHelper.write(modelStoredPath, model);
	}

	public static double testAClassifier(Classifier model, String attributesToBeRemoved, String guidelineName,
			String evaluation) throws Exception {
		String modelStoredPath = filesRootPath + specificFolderPath + guidelineName + "/"
				+ model.getClass().getTypeName();

		// Here is the reference for the following code about getting the
		// probability score
		// http://stackoverflow.com/questions/11960580/weka-classification-likelihood-of-the-classes

		// Deserialize the classifier.
		Classifier classifier = (Classifier) weka.core.SerializationHelper.read(modelStoredPath);

		// Load the test instances.
		Instances originalTestInstances = getTrainingOrTestInstancesForAGuideline(guidelineName, "test");

		Instances attributesDeduplicatedTestInstances = removeDuplicatedAttributes(originalTestInstances,
				guidelineName, "test");

		Instances attributesRemovedTestInstances = removeCertainAttributes(attributesDeduplicatedTestInstances,
				attributesToBeRemoved, guidelineName, "test");

		// Mark the last attribute in each instance as the true class.
		attributesRemovedTestInstances.setClassIndex(attributesRemovedTestInstances.numAttributes() - 1);

		// //convert nominal attributes to binary
		// NominalToBinary nominalToBinary = new NominalToBinary();
		// nominalToBinary.setInputFormat(attributesRemovedTestInstances);
		// attributesRemovedTestInstances =
		// Filter.useFilter(attributesRemovedTestInstances, nominalToBinary);
		//
		// saveInstancesToAnARFF(attributesRemovedTestInstances,
		// "De_Duplicated_Binary", guidelineName,
		// "test");

		// // the following lines is to discretize the data set
		// Discretize filter;
		// // setup filter
		// filter = new Discretize();
		// filter.setInputFormat(attributes_removed_testInstances_0);
		//
		// Instances attributesRemovedTestInstances;
		// // apply filter
		// attributesRemovedTestInstances =
		// Filter.useFilter(attributes_removed_testInstances_0, filter);

		// attributesRemovedTestInstances.setClassIndex(attributesRemovedTestInstances.numAttributes()
		// - 1);

		int totalPositiveCases = getPositiveCaseCountInADatasetInstances(attributesRemovedTestInstances);

		// System.out.println("Total positive cases in this guideline: "
		// + totalPositiveCases);

		// String[][] probability =
		// generateSortedProbabilityArray(classifier,
		// attributesRemovedTestInstances);

		String[][] probability = generateSortedProbabilityArrayForErrorAnalysis(classifier,
				attributesRemovedTestInstances, originalTestInstances);

		// System.out.println("Precision at top 10: "
		// + topKPrecisionWODuplicates(probability, 10, guidelineName));
		//
		// System.out.println("Precision at top 20: "
		// + topKPrecisionWODuplicates(probability, 20, guidelineName));

		// this method doesn't exclude the overlapped guideline citation
		// between training and test
		// System.out.println("Precision at top 10: "
		// + topKPrecisionWDuplicates(probability, 10));
		// System.out.println("Precision at top 20: "
		// + topKPrecisionWDuplicates(probability, 20));

		// // this is the version before error analysis
		// int index = 0;
		// for (final String[] s : probability) {
		// System.out.printf("%10s %4s %4s %5.4f ", s[0], s[1], s[2],
		// Double.valueOf(s[3]));
		// System.out.println("");
		// if (index++ >= 19) {
		// break;
		// }
		// }

		// // this following part is for false positive analysis on the top 20
		// int index = 0;
		// for (final String[] s : probability) {
		// if (s[1].equals("No") && s[2].equals("Yes")) {
		//// System.out.printf("%10s %4s %4s %5.4f %5.4f %5.4f", s[0], s[1],
		// s[2],
		//// Double.valueOf(s[3]), Double.valueOf(s[4]), Double.valueOf(s[5]));
		//
		// System.out.printf("%10s", s[0]);
		// System.out.println("");
		// }
		// if (index++ >= 19) {
		// break;
		// }
		// }

		// // this following part is for false negative analysis on all the
		// // classification results， BUT this method doesn't include
		// // the citations missed by the original PubMed search
		// for (final String[] s : probability) {
		// if (s[1].equals("Yes") && s[2].equals("No")) {
		//// System.out.printf("%10s %4s %4s %5.4f %5.4f %5.4f", s[0], s[1],
		// s[2],
		//// Double.valueOf(s[3]), Double.valueOf(s[4]), Double.valueOf(s[5]));
		//
		// System.out.printf("%10s", s[0]);
		//
		// System.out.println("");
		// }
		// }

		// This part is for false negative analysis on all the
		// classification results. ALSO this method does include
		// the citations missed by the original PubMed search
		LinkedList<String> guidelineCitation = getGuidelineCitationsList(guidelineName);
		LinkedList<String> truePositive = new LinkedList<String>();

		// To get the true positive cases
		for (final String[] s : probability) {
			if (s[1].equals("Yes") && s[2].equals("Yes")) {
				// System.out.printf("%10s %4s %4s %5.4f %5.4f %5.4f", s[0],
				// s[1], s[2], Double.valueOf(s[3]), Double.valueOf(s[4]),
				// Double.valueOf(s[5]));

				truePositive.add(s[0]);

				// System.out.printf("%10s", s[0]);
				// System.out.println("");
			}
		}

		// To get the cases missed out by the algorithm
		for (String s : guidelineCitation) {
			if (!truePositive.contains(s)) {
				System.out.printf(s);
				System.out.println("");
			}
		}

		// here, we may have some issue here for calling this method
		// we can manually evaluate a saved model by the following:
		// https://weka.wikispaces.com/Saving+and+loading+models

		// Evaluation eval = new Evaluation(attributesRemovedTestInstances);
		// eval.evaluateModel(classifier, attributesRemovedTestInstances);

		double[] generalEvaluations = getGeneralEvaluationsOnClassifier(probability, guidelineName, false);

		if (evaluation.equals("topKPrecision")) {

			// System.out.println(guidelineName + " topKPrecision: "
			// + topKPrecisionWODuplicates(probability, 20,
			// guidelineName));

			System.out.println(topKPrecisionWODuplicates(probability, 20, guidelineName));

			return topKPrecisionWODuplicates(probability, 20, guidelineName);
		}

		else if (evaluation.equals("topKMeanAveragePrecision")) {
			// System.out.println(guidelineName + " topKMeanAveragePrecision:
			// "
			// + topKMeanAveragePrecisionWODuplicates(probability, 20,
			// totalPositiveCases, guidelineName));

			System.out.println(
					topKMeanAveragePrecisionWODuplicates(probability, 20, totalPositiveCases, guidelineName));

			return topKMeanAveragePrecisionWODuplicates(probability, 20, totalPositiveCases, guidelineName);
		}

		else if (evaluation.equals("topKMeanReciprocalRank")) {
			// System.out.println(guidelineName + " topKMeanReciprocalRank: "
			// + topKMeanReciprocalRankWODuplicates(probability, 20,
			// guidelineName));

			System.out.println(topKMeanReciprocalRankWODuplicates(probability, 20, guidelineName));

			return topKMeanReciprocalRankWODuplicates(probability, 20, guidelineName);
		}

		// I confirmed the following parts in Weka by printing out several
		// guidelines
		else if (evaluation.equals("Precision")) {

			// System.out.println(guidelineName + " positive precision: "
			// + eval.precision(1));

			// System.out.println(eval.precision(1));
			// return eval.precision(1);

			// System.out.println(generalEvaluations[0]);
			System.out.println();
			return generalEvaluations[0];

		}

		else if (evaluation.equals("Recall")) {

			// System.out.println(guidelineName + " positive recall: "
			// + eval.recall(1));

			// System.out.println(eval.recall(1));
			// return eval.recall(1);

			System.out.println(generalEvaluations[1]);
			return generalEvaluations[1];

		} else if (evaluation.equals("F1")) {

			// System.out.println(guidelineName + " F1: "
			// + eval.fMeasure(1));

			// System.out.println(eval.fMeasure(1));
			// return eval.fMeasure(1);

			System.out.println(generalEvaluations[2]);
			return generalEvaluations[2];
		}

		else {
			System.err.println("Something is wrong: evaluation is not matched!");

			// when it returns 0.00000000, something is wrong!
			return 0.00000000;
		}
	}

	// if there is any column with inappropriate data types from the ARFF files
	// we need to correct them before sending the Weka application right now,
	// I am correcting the column PMID data type from Numeric To Nominal
	public static Instances setCorrectDatatypes(Instances Ins) throws Exception {
		NumericToNominal convert = new NumericToNominal();
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = "1"; // range of variables to make numeric

		convert.setOptions(options);
		convert.setInputFormat(Ins);

		Instances newData = Filter.useFilter(Ins, convert);
		return newData;
	}

	// some attributes are potentially duplicated, which may deteriorate the
	// results, so we need to remove them
	// in this particular case, we need to remove these five:
	// PMID; YearlyScopusCC; YearlyAltmetricScore; Quality; YearsOld

	public static Instances removeDuplicatedAttributes(Instances Ins, String guidelineName, String trainingOrTest)
			throws Exception {

		String[] options = new String[2];
		options[0] = "-R"; // "range"

		options[1] = "1-2, 5, 9, 12";

		Remove remove = new Remove(); // new instance of filter
		remove.setOptions(options); // set options
		remove.setInputFormat(Ins); // inform filter about dataset **AFTER**
									// setting options
		Instances newData = Filter.useFilter(Ins, remove);

		saveInstancesToAnARFF(newData, "De_Duplicated", guidelineName, trainingOrTest);

		return newData;
	}

	// "whatToRemove" can only be exactly same as one of the following:
	// Contains_All; ScopusCC_Removed;...
	public static Instances removeCertainAttributes(Instances Ins, String attributesToBeRemoved,
			String guidelineName, String trainingOrTest) throws Exception {

		if (attributesToBeRemoved.equals("All_Attributes")) {
			return Ins;
		}

		else {
			String[] options = new String[2];
			options[0] = "-R"; // "range"
			if (attributesToBeRemoved.equals("ScopusCC_Removed")) {
				options[1] = "7-8";
			} 
			else if (attributesToBeRemoved.equals("AltmetricScore_Removed")) {
				options[1] = "5-6";
			} 
			else if (attributesToBeRemoved.equals("AltmetricScore_N_ScopusCC_Removed")) {
				options[1] = "5-8";
			}
			else if (attributesToBeRemoved.equals("Only_ScopusCC")) {
				options[1] = "1-6, 9-13";
			}

			// else if
			// (attributesToBeRemoved.equals("Halil_Probability_Removed")) {
			// options[1] = "2";
			// }
			// else if
			// (attributesToBeRemoved.equals("Only_Metadata_Contained")) {
			// options[1] = "4-5, 8-14";
			// }
			// else if (attributesToBeRemoved.equals("JIF_Removed")) {
			// options[1] = "9";
			// }

			else {
				System.err.println("Not desired Columns inputs, check the codes for what can be input!");
				System.exit(0);
			}
			Remove remove = new Remove(); // new instance of filter
			remove.setOptions(options); // set options
			remove.setInputFormat(Ins); // inform filter about dataset **AFTER**
										// setting options
			Instances newData = Filter.useFilter(Ins, remove);

			saveInstancesToAnARFF(newData, attributesToBeRemoved, guidelineName, "");

			return newData;
		}
	}

	public static void saveInstancesToAnARFF(Instances Ins, String whatToRemove, String guidelineName,
			String trainingOrTest) throws IOException {

		// filepath is for the "1+1+9" version
		String filepath = filesRootPath + specificFolderPath + "ARFFs/Attributes-removed ARFFs/" + guidelineName
				+ "_" + whatToRemove + ".arff";
		
//		// filepath is for the previous versions (separate training and test)
//		String filepath = filesRootPath + specificFolderPath + guidelineName + "/" + trainingOrTest + "_"
//				+ whatToRemove + ".arff";

		ArffSaver saver = new ArffSaver();
		saver.setInstances(Ins);

		saver.setFile(new File(filepath));
		saver.writeBatch();
	}

	// to get the number of the positive cases in a Dataset instances
	public static int getPositiveCaseCountInADatasetInstances(Instances inst) {
		int numberOfPositiveCase = 0;

		for (int i = 0; i < inst.numInstances(); i++) {

			if (inst.instance(i).toString(inst.numAttributes() - 1).equals("Yes"))
				numberOfPositiveCase++;
		}

		return numberOfPositiveCase;

	}

	// This method is used for the error analysis: including the original
	// data instance so we can grab the PMID from it

	// Reference for getting the probability score
	// http://stackoverflow.com/questions/11960580/weka-classification-likelihood-of-the-classes

	public static String[][] generateSortedProbabilityArrayForErrorAnalysis(Classifier classifier, Instances inst,
			Instances inst2) throws Exception {

		int numTestInstances = inst.numInstances();

		// String[][] probabilityArray = new String[numTestInstances][4];

		// I am temporarily changing this array to 6 columns so that it can
		// contain more columns to identify the PMIDs that will be used for the
		// error analysis
		String[][] probabilityArray = new String[numTestInstances][6];

		// Loop over each test instance.
		for (int i = 0; i < numTestInstances; i++) {
			// Get the true class label from the instance's own classIndex.
			String trueClassLabel = inst.instance(i).toString(inst.classIndex());

			// Make the prediction here.
			double predictionIndex = classifier.classifyInstance(inst.instance(i));

			// Get the predicted class label from the predictionIndex.
			String predictedClassLabel = inst.classAttribute().value((int) predictionIndex);

			// Get the prediction probability distribution.
			double[] predictionDistribution = classifier.distributionForInstance(inst.instance(i));

			// Loop over all the prediction labels in the distribution.
			for (int predDistIndex = 0; predDistIndex < predictionDistribution.length; predDistIndex++) {

				// // Get this distribution index's class label.
				// String predictionDistributionIndexAsClassLabel = inst
				// .classAttribute().value(predictionDistributionIndex);
				//
				// // Get the probability.
				// double predictionProbability =
				// predictionDistribution[predictionDistributionIndex];

				// this part is for the error analysis
				probabilityArray[i][0] = inst2.instance(i).toString(inst2.instance(i).attribute(0));
				// probabilityArray[i][0] = "" + (i+1);
				probabilityArray[i][1] = trueClassLabel;

				probabilityArray[i][2] = predictedClassLabel;

				probabilityArray[i][3] = Double.toString(predictionDistribution[1]);

				// these two columns are added for identifying the PMIDs in the
				// error analysis
				probabilityArray[i][4] = inst.instance(i).toString(inst.instance(i).attribute(1));

				probabilityArray[i][5] = inst.instance(i).toString(inst.instance(i).attribute(6));

				// probability[i][3].format("%.2f");

				// I want to figure out how to populate 2-D array row-by-row
				// probability[i] =
				// {testInstances.instance(i).toString(testInstances.instance(i).attribute(0)),
				// trueClassLabel,
				// predictedClassLabel,
				// Double.toString(predictionDistribution[1]}
			}
			// System.out.printf("\n");
		}

		// System.out.println(Arrays.deepToString(probabilityArray));

		// I got the following codes by editing this:
		// http://stackoverflow.com/questions/4907683/sort-a-two-dimensional-array-based-on-one-column
		// I need to look into the Java comparator part for the understanding
		Arrays.sort(probabilityArray, new Comparator<String[]>() {

			public int compare(final String[] entry1, final String[] entry2) {
				final Double probValue1 = Double.valueOf(entry1[3]);
				final Double probValue2 = Double.valueOf(entry2[3]);
				// I need to order from high to low, so I use this one
				return probValue2.compareTo(probValue1);
			}
		});

		return probabilityArray;
	}

	// Reference for getting the probability score
	// http://stackoverflow.com/questions/11960580/weka-classification-likelihood-of-the-classes
	public static String[][] generateSortedProbabilityArray(Classifier classifier, Instances inst) throws Exception {

		int numTestInstances = inst.numInstances();

		// String[][] probabilityArray = new String[numTestInstances][4];

		// I am temporarily changing this array to 6 columns so that it can
		// contain more columns to identify the PMIDs that will be used for the
		// error analysis
		String[][] probabilityArray = new String[numTestInstances][6];

		// Loop over each test instance.
		for (int i = 0; i < numTestInstances; i++) {
			// Get the true class label from the instance's own classIndex.
			String trueClassLabel = inst.instance(i).toString(inst.classIndex());

			// Make the prediction here.
			double predictionIndex = classifier.classifyInstance(inst.instance(i));

			// Get the predicted class label from the predictionIndex.
			String predictedClassLabel = inst.classAttribute().value((int) predictionIndex);

			probabilityArray[i][0] = "" + (i + 1);
			probabilityArray[i][1] = trueClassLabel;
			probabilityArray[i][2] = predictedClassLabel;			
				
			
			// Get the prediction probability distribution.
			double[] predictionDistribution = classifier.distributionForInstance(inst.instance(i));

			// if "@attribute Guideline {No,Yes}
			// then we should use "predictionDistribution[1]".
			// else if "@attribute Guideline {Yes,No}", use
			// "predictionDistribution[0]".
			// so all depend on the index of "Yes"
			probabilityArray[i][3] = Double.toString(predictionDistribution[0]);
			
			
//			// if we have the PMID attribute, these two columns are added for
//			// identifying the PMIDs in the error analysis
//			probabilityArray[i][4] = inst.instance(i).toString(inst.instance(i).attribute(1));
//			probabilityArray[i][5] = inst.instance(i).toString(inst.instance(i).attribute(6));
			
//			// this part is not needed for ranking the results,
//			// but I used for confirming the GUI results
//			// Loop over all the prediction labels in the distribution.
//			for (int predDistIndex = 0; predDistIndex < predictionDistribution.length; predDistIndex++) {
//
//				// Get this distribution index's class label.
//				String predictionDistributionIndexAsClassLabel = inst.classAttribute().value(predDistIndex);
//
//				// Get the probability.
//				double predictionProbability = predictionDistribution[predDistIndex];
//
//				System.out.printf("[%3s : %15.14f]", predictionDistributionIndexAsClassLabel, predictionProbability);
//			}
//			// System.out.println();

		}
		
		// System.out.println(Arrays.deepToString(probabilityArray));
		
		//// this line will be used to print out a 2-d array for debug purpose
		//// http://stackoverflow.com/questions/409784/whats-the-simplest-way-to-print-a-java-array
//		System.out.println(
//				Arrays.deepToString(probabilityArray).replaceAll("],", "]," + System.getProperty("line.separator")));
//		Thread.sleep(1000000);

		// I got the following codes by editing this:
		// http://stackoverflow.com/questions/4907683/sort-a-two-dimensional-array-based-on-one-column
		// I need to look into the Java comparator part for the understanding
		Arrays.sort(probabilityArray, new Comparator<String[]>() {

			public int compare(final String[] entry1, final String[] entry2) {
				final Double probValue1 = Double.valueOf(entry1[3]);
				final Double probValue2 = Double.valueOf(entry2[3]);
				// I need to order from high to low, so I use this one
				return probValue2.compareTo(probValue1);

				// // If I need to order from low to high, I use this one
				// return probValue1.compareTo(probValue2);
			}
		});
		
//		System.out.println();
//		System.out.println();
//		System.out.println(
//				Arrays.deepToString(probabilityArray).replaceAll("],", "]," + System.getProperty("line.separator")));
//		Thread.sleep(1000000);
		
		return probabilityArray;
	}

	
	/**
	 * Used to get generalEvaluationsOnClassifier with two duplicate versions
	 * @param probRankedArray
	 * @param guidelineName
	 * @param duplicatesOK
	 *            [True]: Overlap doesn't matter (includes the overlap guidelines between training and
	 *            test PMIDs). [False]: excludes the overlap guidelines between
	 *            training and test PMIDs
	 * @return
	 * @throws Exception
	 */
	public static double[] getGeneralEvaluationsOnClassifier(String[][] probRankedArray,
			String guidelineName, boolean duplicatesOK) throws Exception {
		
		int truePositiveCount = 0;
		int predictedPositiveCount = 0;
		
		if (duplicatesOK){
			for (int i = 0; i < probRankedArray.length; i++) {
				if (probRankedArray[i][2].equals("Yes")) {
					predictedPositiveCount++;

					if (probRankedArray[i][1].equals("Yes") ) {
						truePositiveCount++;
					}
				}
			}			
		}
		
		else if (!duplicatesOK){
			String overlapFilepath = filesRootPath + specificFolderPath + guidelineName + "/" + "overlap.txt";
			// Call a method from the HighImpactPubMedArticleRanking class
			LinkedList<String> overlap = HighImpactPubMedArticleRanking.getLinkedListFromAFile(overlapFilepath);

			for (int i = 0; i < probRankedArray.length; i++) {
				if (probRankedArray[i][2].equals("Yes")) {
					predictedPositiveCount++;

					if (probRankedArray[i][1].equals("Yes") && !overlap.contains(probRankedArray[i][0])) {
						truePositiveCount++;
					}
				}
			}
			
		}
		


		double[] genEvalsOnClassifier = new double[3];
		// System.out.println("top K Precision:" + (double) YesCounts / K);
		// Convert this YesCounts to (double), otherwise it will return 0;

		// System.out.println(truePositiveCount + ": " + predictedPositiveCount
		// + ": "
		// + getGuidelineCitationsList(guidelineName).size());

		//// this version has a bug if we use predictedPositiveCount to tell the
		//// value,
		//// we should use truePositiveCount for telling the value;

		// // if the predictedPositiveCount is 0, then truePositiveValue must be
		// zero,
		// // then the precision will be assigned a value of 0;
		// genEvalsOnClassifier[0] = (predictedPositiveCount == 0 ? 0:
		// (double) truePositiveCount / predictedPositiveCount);

		// genEvalsOnClassifier[1] = (double) truePositiveCount /
		// getGuidelineCitationsList(guidelineName).size();
		//
		// // if the predictedPositiveCount is 0, then truePositiveValue must be
		// zero,
		// // then the precision will be assigned a value of 0;
		// genEvalsOnClassifier[2] = (predictedPositiveCount == 0 ? 0:
		// (double) 2 * genEvalsOnClassifier[0] * genEvalsOnClassifier[1]
		// / (genEvalsOnClassifier[0] + genEvalsOnClassifier[1]));

		// if the truePositiveCount is 0, then the precision will be assigned a
		// value of 0;
		genEvalsOnClassifier[0] = (truePositiveCount == 0 ? 0 : (double) truePositiveCount / predictedPositiveCount);

		genEvalsOnClassifier[1] = (truePositiveCount == 0 ? 0
				: (double) truePositiveCount / getGuidelineCitationsList(guidelineName).size());

		genEvalsOnClassifier[2] = (truePositiveCount == 0 ? 0
				: (double) 2 * genEvalsOnClassifier[0] * genEvalsOnClassifier[1]
						/ (genEvalsOnClassifier[0] + genEvalsOnClassifier[1]));

		return genEvalsOnClassifier;

	}

	// this version doesn't exclude the overlap guidelines between training and
	// test PMIDs from the top-K
	public static double topKPrecisionWDuplicates(String[][] probRankedArray, int K) {
		if (probRankedArray.length < K) {
			System.err.println("The total number of instances: " + probRankedArray.length + " is less than: " + K);
			System.exit(0);
		}
		int YesCounts = 0;
		for (int i = 0; i < K; i++) {
			if (probRankedArray[i][1].equals("Yes")) {
				YesCounts++;
			}
		}

//		 System.out.println("Yes: " + YesCounts);

//		System.out.println((double) YesCounts / K);
		// Convert this YesCounts to (double), otherwise it will return 0;
		return (double) YesCounts / K;
	}

	// this version excludes the overlap guidelines between training and test
	// PMIDs from the top-K
	public static double topKPrecisionWODuplicates(String[][] probRankedArray, int K, String guidelineName)
			throws Exception {
		String overlapFilepath = filesRootPath + specificFolderPath + guidelineName + "/" + "overlap.txt";
		// call the existing method from the HighImpactPubMedArticleRanking
		// class
		LinkedList<String> overlap = HighImpactPubMedArticleRanking.getLinkedListFromAFile(overlapFilepath);

		if (probRankedArray.length < K) {
			System.err.println("The total number of instances: " + probRankedArray.length + " is less than: " + K);
			System.exit(0);
		}
		int YesCounts = 0;
		for (int i = 0; i < K; i++) {
			if (probRankedArray[i][1].equals("Yes") && !overlap.contains(probRankedArray[i][0])) {
				YesCounts++;
			}
		}
		// System.out.println("top K Precision:" + (double) YesCounts / K);
		// Convert this YesCounts to (double), otherwise it will return 0;
		return (double) YesCounts / K;
	}

	// the following codes were used to get the count of each guideline
	// citations, so that the recall will be calculated this way

	public static LinkedList<String> getGuidelineCitationsList(String guidelineName) throws Exception {

		String guidelineCitationFilepath = "E:/Studying/Box Sync/workspace/Thesis/RCTs_N_SRs_PMIDs/" + guidelineName
				+ ".txt";

		LinkedList<String> gcll = HighImpactPubMedArticleRanking.getLinkedListFromAFile(guidelineCitationFilepath);

		return gcll;
		// return gcll.size();
	}

	// the "genEval" must match one of the exact strings: "Precision",
	// "Recall", "F1"
	public static double[] getGeneralEvaluationsOnHalilScore(String[][] probRankedArray,
			int positiveSamplesCount, int totalCases, String guidelineName) throws Exception {

		// index 0 for Precision; index 1 for Recall; index 2 for F1;
		double[] genEvals = new double[9];

		int halilHighQualityCount = 0;

		int truePositive = 0;

		int trueNegative = 0;

		for (int i = 0; i < probRankedArray.length; i++) {
			if (Double.valueOf(probRankedArray[i][3]) == 1) {
				halilHighQualityCount++;
			}

			if (probRankedArray[i][1].equals("Yes") && Double.valueOf(probRankedArray[i][3]) == 1) {
				truePositive++;
			}

			if (probRankedArray[i][1].equals("No") && Double.valueOf(probRankedArray[i][3]) == 0) {
				trueNegative++;
			}
		}

		// index 0 for positive Precision;
		// index 1 for positive Recall;
		// index 2 for positive F1;
		genEvals[0] = (double) truePositive / halilHighQualityCount;
		// genEvals[1] = (double) truePositive / positiveSamplesCount;
		genEvals[1] = (double) truePositive / getGuidelineCitationsList(guidelineName).size();
		genEvals[2] = (double) 2 * genEvals[0] * genEvals[1] / (genEvals[0] + genEvals[1]);

		// index 3 for negative Precision;
		// index 4 for negative Recall;
		// index 5 for negative F1;
		genEvals[3] = (double) trueNegative / (totalCases - halilHighQualityCount);
		genEvals[4] = (double) trueNegative / (totalCases - positiveSamplesCount);
		genEvals[5] = (double) 2 * genEvals[3] * genEvals[4] / (genEvals[3] + genEvals[4]);

		// index 6 for weighted Precision;
		// index 7 for weighted Recall;
		// index 8 for weighted F1;
		genEvals[6] = (double) (positiveSamplesCount * genEvals[0]
				+ (totalCases - positiveSamplesCount) * genEvals[3]) / totalCases;
		genEvals[7] = (double) (positiveSamplesCount * genEvals[1]
				+ (totalCases - positiveSamplesCount) * genEvals[4]) / totalCases;
		genEvals[8] = (double) (positiveSamplesCount * genEvals[2]
				+ (totalCases - positiveSamplesCount) * genEvals[5]) / totalCases;

		return genEvals;
	}

	// this version doesn't exclude the overlap (there is no overlap. e.g., 10
	// fold cross validation in a data set)
	// guidelines between training and test PMIDs from the top-K
	public static double topKMeanAveragePrecisionWDuplicates(String[][] probRankedArray, int K,
			int positiveSamplesCount) throws Exception {

		int denominator = K < positiveSamplesCount ? K : positiveSamplesCount;

		if (probRankedArray.length < K) {
			System.err.println("The total number of instances: " + probRankedArray.length + " is less than: " + K);
			System.exit(0);
		}

		double averagePrecision = 0.000;

		for (int i = 0, YesCounts = 0; i < K; i++) {
			if (probRankedArray[i][1].equals("Yes")) {
				YesCounts++;
				averagePrecision = averagePrecision + (double) (YesCounts) / (i + 1);
				// System.out.println(YesCounts + " " + (i + 1) + " "
				// + averagePrecision + " " + averagePrecision
				// / denominator);
			}
		}

//		System.out.println((double) averagePrecision / denominator);

		return (double) averagePrecision / denominator;
	}

	// this version excludes the overlap guidelines between training and test
	// PMIDs from the top-K
	public static double topKMeanAveragePrecisionWODuplicates(String[][] probRankedArray, int K,
			int positiveSamplesCount, String guidelineName) throws Exception {
		String overlapFilepath = filesRootPath + specificFolderPath + guidelineName + "/" + "overlap.txt";
		// call the existing method from the HighImpactPubMedArticleRanking
		// class
		LinkedList<String> overlap = HighImpactPubMedArticleRanking.getLinkedListFromAFile(overlapFilepath);

		int denominator = K < positiveSamplesCount ? K : positiveSamplesCount;

		if (probRankedArray.length < K) {
			System.err.println("The total number of instances: " + probRankedArray.length + " is less than: " + K);
			System.exit(0);
		}

		double averagePrecision = 0.000;

		for (int i = 0, YesCounts = 0; i < K; i++) {
			if (probRankedArray[i][1].equals("Yes") && !overlap.contains(probRankedArray[i][0])) {
				YesCounts++;
				averagePrecision = averagePrecision + (double) (YesCounts) / (i + 1);
				// System.out.println(YesCounts + " " + (i + 1) + " "
				// + averagePrecision + " " + averagePrecision
				// / denominator);
			}
		}

		return (double) averagePrecision / denominator;
	}

	// this version doesn't exclude the overlap (there is no overlap. e.g., 10
	// fold cross validation in a data set)
	// guidelines between training and test PMIDs from the top-K

	// http://link.springer.com/referenceworkentry/10.1007%2F978-0-387-39940-9_488
	// https://en.wikipedia.org/wiki/Mean_reciprocal_rank
	public static double topKMeanReciprocalRankWDuplicates(String[][] probRankedArray, int K) throws Exception {

		if (probRankedArray.length < K) {
			System.err.println("The total number of instances: " + probRankedArray.length + " is less than: " + K);
			System.exit(0);
		}

		for (int i = 0; i < K; i++) {
			if (probRankedArray[i][1].equals("Yes")) {
				// System.out.println("First Positive Sample Position: " + (i +
				// 1)
				// + " " + (double) 1 / (i + 1));
//				System.out.println((double) 1 / (i + 1));
				return (double) 1 / (i + 1);
			}
		}

		// if the first positive sample doesn't exist in the top K,
		// return 0.00 for this measure;
		return 0.000;
	}

	// this version excludes the overlap guidelines between training and test
	// PMIDs from the top-K
	// http://link.springer.com/referenceworkentry/10.1007%2F978-0-387-39940-9_488
	// https://en.wikipedia.org/wiki/Mean_reciprocal_rank
	public static double topKMeanReciprocalRankWODuplicates(String[][] probRankedArray, int K,
			String guidelineName) throws Exception {
		String overlapFilepath = filesRootPath + specificFolderPath + guidelineName + "/" + "overlap.txt";
		// call the existing method from the HighImpactPubMedArticleRanking
		// class
		LinkedList<String> overlap = HighImpactPubMedArticleRanking.getLinkedListFromAFile(overlapFilepath);

		if (probRankedArray.length < K) {
			System.err.println("The total number of instances: " + probRankedArray.length + " is less than: " + K);
			System.exit(0);
		}

		for (int i = 0; i < K; i++) {
			if (probRankedArray[i][1].equals("Yes") && !overlap.contains(probRankedArray[i][0])) {
				// System.out.println("First Positive Sample Position: " + (i +
				// 1)
				// + " " + (double) 1 / (i + 1));
				// System.out.println();
				return (double) 1 / (i + 1);
			}
		}

		// if the first positive sample doesn't exist in the top K,
		// return 0.00 for this measure;
		return 0.000;
	}

	public static void getOverlapForThePositivesBetweenTrainingNTest(String guidelineName) throws Exception {

		LinkedList<String> trainingLL = getPositiveSamplesForAnArffDataset(guidelineName, "training");

		LinkedList<String> testLL = getPositiveSamplesForAnArffDataset(guidelineName, "test");

		LinkedList<String> overlapLL = trainingLL;

		overlapLL.retainAll(testLL);

		String filepath = filesRootPath + specificFolderPath + guidelineName + "/" + "overlap" + ".txt";

		FileWriter fw = new FileWriter(filepath, false);

		BufferedWriter bw = new BufferedWriter(fw);

		for (int i = 0; i < overlapLL.size(); i++) {

			bw.write(overlapLL.get(i) + "\r\n");
		}

		bw.close();
		fw.close();
		System.out.println(guidelineName + " overlap is done!");
	}

	public static LinkedList<String> getPositiveSamplesForAnArffDataset(String guidelineName,
			String trainingOrTest) throws Exception {
		Instances testInstances = getTrainingOrTestInstancesForAGuideline(guidelineName, trainingOrTest);
		LinkedList<String> ll = new LinkedList<String>();
		for (int i = testInstances.numInstances() - 1; i >= 0; i--) {
			if (testInstances.instance(i)
					.toString(testInstances.instance(i).attribute(testInstances.numAttributes() - 1)).equals("Yes")) {
				ll.addLast(testInstances.instance(i).toString(testInstances.instance(i).attribute(0)));
			}
		}

		return ll;
	}


	public static void getEvaluationsOnHalilScore(int k) throws Exception {

		for (int eval = 0; eval < evaluations.length; eval++) {
			
			System.out.println(evaluations[eval]);
			// this one needs to average over all the guidelines
			for (int i = 0; i < guidelines.length; i++) {
				System.out.print(guidelines[i] + ": ");

				precisionsArray[i] = getEvaluationsOnHalilScoreForAGuideline(guidelines[i], k,
						evaluations[eval]);

				// System.out.println();
			}

			Statistics sta = new Statistics(precisionsArray);
			// System.out.print("Mean: ");
			System.out.printf("%3.2f", sta.getMean());
			System.out.printf("±");
			// System.out.print("SD: ");
			System.out.printf("%3.2f", sta.getStdDev());
			System.out.println("\n");
			// System.out.println("Median: " + sta.median() + "\n\n\n");
		}
	}

	
	// This method is used to generate the probability array for the new
	// baseline that is relevance sort, the array items are ordered
	// according to the search results based on relevance sort filter
	// on the display setting, and the array looks like PMID, "yes"/"No",
	// depends on whether the citation is in the guideline
	public static String[][] generateProbabilityArrayForRelevanceSort(String guidelineName) throws Exception {

		String relevancesortFilepath = "E:/Studying/Box Sync/workspace/Thesis/DataSet_PMIDs/" + guidelineName
				+ ".txt";

		String guidelinecitationsFilepath = "E:/Studying/Box Sync/workspace/Thesis/RCTs_N_SRs_PMIDs/" + guidelineName
				+ ".txt";

		LinkedList<String> rsll = HighImpactPubMedArticleRanking.getLinkedListFromAFile(relevancesortFilepath);

		int rsllSize = rsll.size();
		String[][] probArray = new String[rsllSize][2];

		LinkedList<String> gcll = HighImpactPubMedArticleRanking.getLinkedListFromAFile(guidelinecitationsFilepath);
		for (int i = 0; i < rsllSize; i++) {
			probArray[i][0] = rsll.get(i);
			probArray[i][1] = gcll.contains(rsll.get(i)) ? "Yes" : "No";
		}

		// // print out for the testing purpose;
		// for (int i = 0; i < probArray.length; i++) {
		// System.out.println(probArray[i][0]+" "+ probArray[i][1]);
		// }

		return probArray;
	}

	public static double getEvaluationsOnRelevanceSortForAGuideline(String guidelineName, int k,
			String eval) throws Exception {

		String[][] probArray = generateProbabilityArrayForRelevanceSort(guidelineName);

		int totalYes = 0;

		for (int i = 0; i < probArray.length; i++) {
			if (probArray[i][1].equals("Yes")) {
				totalYes++;
			}
		}

		double relevanceSortPrecision = (double) totalYes / probArray.length;

		double relevanceSortRecall = (double) totalYes / getGuidelineCitationsList(guidelineName).size();

		double relevanceSortF1 = (double) 2 * relevanceSortPrecision * relevanceSortRecall
				/ (relevanceSortPrecision + relevanceSortRecall);

		if (eval.equals("topKPrecision")) {
			// System.out.println(guidelineName + " Relevance_Sort Top K
			// Precision: "
			// + topKPrecisionWODuplicates(probability, k,
			// guidelineName));
			// System.out.println(topKPrecisionWODuplicates(probArray, k,
			// guidelineName));
			// return topKPrecisionWODuplicates(probArray, k,
			// guidelineName);

			//// baseline doesn't need to exclude duplicate
			//// because it is not based on the training data
			System.out.println(topKPrecisionWDuplicates(probArray, k));
			return topKPrecisionWDuplicates(probArray, k);
		}

		else if (eval.equals("topKMeanAveragePrecision")) {
			// System.out.println(guidelineName + " Relevance_Sort
			// topKMeanAveragePrecision: "
			// + topKMeanAveragePrecisionWODuplicates(probability, k,
			// totalPositiveCases, guidelineName));
			// System.out.println(topKMeanAveragePrecisionWODuplicates(probArray,
			// k,
			// totalYes, guidelineName));
			// return topKMeanAveragePrecisionWODuplicates(probArray, k,
			// totalYes, guidelineName);

			//// baseline doesn't need to exclude duplicate
			//// because it is not based on the training data
			System.out.println(topKMeanAveragePrecisionWDuplicates(probArray, k, totalYes));
			return topKMeanAveragePrecisionWDuplicates(probArray, k, totalYes);
		}

		else if (eval.equals("topKMeanReciprocalRank")) {
			// System.out.println(guidelineName + " Relevance_Sort
			// topKMeanReciprocalRank: "
			// + topKMeanReciprocalRankWODuplicates(probability, k,
			// guidelineName));
			// System.out.println(topKMeanReciprocalRankWODuplicates(probArray,
			// k,
			// guidelineName));
			// return topKMeanReciprocalRankWODuplicates(probArray, k,
			// guidelineName);

			//// baseline doesn't need to exclude duplicate
			//// because it is not based on the training data

			System.out.println(topKMeanReciprocalRankWDuplicates(probArray, k));
			return topKMeanReciprocalRankWDuplicates(probArray, k);
		}

		else if (eval.equals("Precision")) {
			// System.out.println(guidelineName + " relevanceSortPrecision: "
			// + relevanceSortPrecision);
			System.out.println(relevanceSortPrecision);
			return relevanceSortPrecision;
		}

		else if (eval.equals("Recall")) {
			// System.out.println(guidelineName + " relevanceSortRecall: " +
			// relevanceSortRecall);
			System.out.println(relevanceSortRecall);
			return relevanceSortRecall;
		}

		else if (eval.equals("F1")) {
			// System.out.println(guidelineName + " relevanceSortF1: " +
			// relevanceSortF1);
			System.out.println(relevanceSortF1);
			return relevanceSortF1;
		}

		// when it returns 0.00000000, something is wrong!
		return 0.00000000;

	}

	// this one needs to average over all the guidelines
	public static void getEvaluationsOnRelevanceSort(int k) throws Exception {
		for (int eval = 0; eval < evaluations.length; eval++) {
			System.out.println(evaluations[eval]);

			// Averaging over all the guidelines
			for (int i = 0; i < guidelines.length; i++) {
				System.out.print(guidelines[i] + ": ");
				precisionsArray[i] = getEvaluationsOnRelevanceSortForAGuideline(guidelines[i], k,
						evaluations[eval]);
				// System.out.println();
			}

			Statistics sta = new Statistics(precisionsArray);

			// System.out.print("Mean: ");
			System.out.printf("%3.2f", sta.getMean());
			System.out.printf("±");
			// System.out.print("SD: ");
			System.out.printf("%3.2f", sta.getStdDev());
			System.out.println("\n");

			// System.out.println("Median: " + sta.median() + "\n\n\n");
		}
	}

	
	// Calculate the topK precision score based on all the test data sets
	// by ranking the instances by the probability scores from high to low,
	// then take the top K (e.g., 20) to check how many "Yes" within it
	// BUT use the very original data set with PMID, prob, quality included
	public static double getEvaluationsOnHalilScoreForAGuideline(String guidelineName, int k, String eval)
			throws Exception {

		// // Load the test instances. we may need to change the file path every
		// // time as it got changed frequently
		// Instances testInstances =
		// getTrainingOrTestInstancesForAGuideline(guidelineName,
		// "test");
		

		// use the path containing the very original data set with PMID, prob,
		// quality included, so that the predicted "Yes" will work
		// for "1+1+9/" evaluation on Halil Score to generate all columns
		String filename = filesRootPath + specificFolderPath 
						+ "ARFFs/Previous ARFFs/Step 3_ ARFFs with duplicated attributes/" + guidelineName + ".arff";
		
//		String filename = filesRootPath + specificFolderPath 
//						+ "ARFFs/" + guidelineName + ".arff";
		
//		System.out.println(filename);
		DataSource source = new DataSource(filename);	
		Instances testInstances = source.getDataSet();		
		
		// Mark the last attribute in each instance as the true class.
		testInstances.setClassIndex(testInstances.numAttributes() - 1);

		int totalPositiveCases = getPositiveCaseCountInADatasetInstances(testInstances);

		int numTestInstances = testInstances.numInstances();

		// System.out.printf("There are %d test instances\n", numTestInstances);

		String[][] probability = new String[numTestInstances][4];

		// Loop over each test instance.
		for (int i = 0; i < numTestInstances; i++) {
			// Get the true class label from the instance's own classIndex.
			String trueClassLabel = testInstances.instance(i).toString(testInstances.classIndex());

			// PMID
			probability[i][0] = testInstances.instance(i).toString(testInstances.instance(i).attribute(0));

			probability[i][1] = trueClassLabel;

			// Halil Probability Score
			probability[i][2] = testInstances.instance(i).toString(testInstances.instance(i).attribute(3));

			// Halil Probability Quality Value
			// This data is collected particularly for calculating the precision
			// recall and F1 score for the Halil probability score
			probability[i][3] = testInstances.instance(i).toString(testInstances.instance(i).attribute(4));

		}

		// I got the following codes by editing this:
		// http://stackoverflow.com/questions/4907683/sort-a-two-dimensional-array-based-on-one-column
		// I need to look into the Java comparator part for the understanding
		Arrays.sort(probability, new Comparator<String[]>() {
			public int compare(final String[] entry1, final String[] entry2) {
				final Double probValue1 = Double.valueOf(entry1[2]);
				final Double probValue2 = Double.valueOf(entry2[2]);
				// I need to order from high to low, so I use this one
				return probValue2.compareTo(probValue1);
			}
		});

//		// Confirm that the results are correct
//		for (int i = 0; i < k; i++) {
//			System.out.println(
//					probability[i][0] + " " + probability[i][1] + " " + probability[i][2] + " " + probability[i][3]);
//		}

//		Thread.sleep(1000000);
		
		double[] genEval = getGeneralEvaluationsOnHalilScore(probability, totalPositiveCases, numTestInstances,
				guidelineName);
		// System.out.println(guidelineName + " halilPrecision: " +
		// genEval[0]);
		double halilPrecision = genEval[0];
		// System.out.println(guidelineName + " halilRecall: " + genEval[1]);
		double halilRecall = genEval[1];
		// System.out.println(guidelineName + " F1: " + genEval[2]);
		double halilF1 = genEval[2];

		if (eval.equals("topKPrecision")) {
			// System.out.println(guidelineName + " Halil Top K Precision: "
			// + topKPrecisionWODuplicates(probability, 20,
			// guidelineName));
			// System.out.println(topKPrecisionWODuplicates(probability, 20,
			// guidelineName));
			// return topKPrecisionWODuplicates(probability, 20,
			// guidelineName);

			//// baseline doesn't need to exclude duplicate
			//// because it is not based on the training data

			System.out.println(topKPrecisionWDuplicates(probability, k));
			return topKPrecisionWDuplicates(probability, k);
		}

		else if (eval.equals("topKMeanAveragePrecision")) {
			// System.out.println(guidelineName + " Halil
			// topKMeanAveragePrecision: "
			// + topKMeanAveragePrecisionWODuplicates(probability, 20,
			// totalPositiveCases, guidelineName));
			// System.out.println(topKMeanAveragePrecisionWODuplicates(probability,
			// 20,
			// totalPositiveCases, guidelineName));
			// return topKMeanAveragePrecisionWODuplicates(probability, 20,
			// totalPositiveCases, guidelineName);

			//// baseline doesn't need to exclude duplicate
			//// because it is not based on the training data

			System.out.println(topKMeanAveragePrecisionWDuplicates(probability, k, totalPositiveCases));
			return topKMeanAveragePrecisionWDuplicates(probability, k, totalPositiveCases);
		}

		else if (eval.equals("topKMeanReciprocalRank")) {
			// System.out.println(guidelineName + " Halil
			// topKMeanReciprocalRank: "
			// + topKMeanReciprocalRankWODuplicates(probability, 20,
			// guidelineName));
			// System.out.println(topKMeanReciprocalRankWODuplicates(probability,
			// 20,
			// guidelineName));
			// return topKMeanReciprocalRankWODuplicates(probability, 20,
			// guidelineName);

			//// baseline doesn't need to exclude duplicate
			//// because it is not based on the training data

			System.out.println(topKMeanReciprocalRankWDuplicates(probability, k));
			return topKMeanReciprocalRankWDuplicates(probability, k);

		}

		else if (eval.equals("Precision")) {
			// System.out.println(guidelineName + " halilPrecision: " +
			// genEval[0]);
			System.out.println(genEval[0]);
			return halilPrecision;
		}

		else if (eval.equals("Recall")) {
			// System.out.println(guidelineName + " halilRecall: " +
			// genEval[1]);
			System.out.println(genEval[1]);
			return halilRecall;
		}

		else if (eval.equals("F1")) {
			// System.out.println(guidelineName + " F1: " + genEval[2]);
			System.out.println(genEval[2]);
			return halilF1;
		}

		// when it returns 0.00000000, something is wrong!
		return 0.00000000;

		// System.out.println(topKPrecisionWDuplicates(probability, k));
		// // for right now, just work on the top 10 precision
		// return topKPrecisionWDuplicates(probability, k);

		// for (final String[] s : probability) {
		// System.out.printf("%10s %4s %4s %5.4f ", s[0], s[1], s[2],
		// Double.valueOf(s[3]));
		// System.out.println("");
		// }
	}

	public static void printOutAllWekaClassifiers() {
		// the source for all the classifiers
		// http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html
		String allClassifiersInAString = "AbstractClassifier, AdaBoostM1, AdditiveRegression, AttributeSelectedClassifier, Bagging, BayesNet, BayesNetGenerator, BIFReader, ClassificationViaRegression, CostSensitiveClassifier, CVParameterSelection, DecisionStump, DecisionTable, EditableBayesNet, FilteredClassifier, GaussianProcesses, GeneralRegression, HoeffdingTree, IBk, InputMappedClassifier, IteratedSingleClassifierEnhancer, IterativeClassifierOptimizer, J48, JRip, KStar, LinearRegression, LMT, LMTNode, Logistic, LogisticBase, LogitBoost, LWL, M5Base, M5P, M5Rules, MultiClassClassifier, MultiClassClassifierUpdateable, MultilayerPerceptron, MultipleClassifiersCombiner, MultiScheme, NaiveBayes, NaiveBayesMultinomial, NaiveBayesMultinomialText, NaiveBayesMultinomialUpdateable, NaiveBayesUpdateable, NeuralNetwork, OneR, ParallelIteratedSingleClassifierEnhancer, ParallelMultipleClassifiersCombiner, PART, PMMLClassifier, PreConstructedLinearModel, RandomCommittee, RandomForest, RandomizableClassifier, RandomizableFilteredClassifier, RandomizableIteratedSingleClassifierEnhancer, RandomizableMultipleClassifiersCombiner, RandomizableParallelIteratedSingleClassifierEnhancer, RandomizableParallelMultipleClassifiersCombiner, RandomizableSingleClassifierEnhancer, RandomSubSpace, RandomTree, Regression, RegressionByDiscretization, REPTree, RuleNode, RuleSetModel, SerializedClassifier, SGD, SGDText, SimpleLinearRegression, SimpleLogistic, SingleClassifierEnhancer, SMO, SMOreg, Stacking, SupportVectorMachineModel, TreeModel, Vote, VotedPerceptron, ZeroR";

		String[] str2 = allClassifiersInAString.split(", ");

		for (int i = 0; i < str2.length; i++) {
			System.out.println("new " + str2[i] + "(),");
		}
	}

}

//// =====>the following part is used for the hyper-parameter
//// =====>optimization for the all the chosen classifiers

// the following codes will be used to initiate the hyper parameters that I
// got from running Auto-Weka manually for each individual classifier

// BayesNet bn = new BayesNet();
//// TAN tan = new TAN();
//// bn.setSearchAlgorithm(tan);
//// bn.setUseADTree(false);
// classifiers.add(bn);

// BayesNet bn_X2 = new BayesNet();
//// -_0__wcbbn_00_D REMOVED
//// -_0__wcbbn_01_Q weka.classifiers.bayes.net.search.local.HillClimber
// HillClimber hc = new HillClimber();
// bn_X2.setSearchAlgorithm(hc);
// bn_X2.setUseADTree(false);
// classifiers.add(bn_X2);

// BayesNet bn_X3 = new BayesNet();
//// -_0__wcbbn_00_D REMOVED
//// -_0__wcbbn_01_Q weka.classifiers.bayes.net.search.local.SimulatedAnnealing
// SimulatedAnnealing sa = new SimulatedAnnealing();
// bn_X3.setSearchAlgorithm(sa);
// bn_X3.setUseADTree(false);
// classifiers.add(bn_X3);

// BayesNet bn_X4 = new BayesNet();
//// -_0__wcbbn_00_D REMOVE_PREV
//// -_0__wcbbn_01_Q weka.classifiers.bayes.net.search.local.TabuSearch
// TabuSearch ts = new TabuSearch();
// bn_X4.setSearchAlgorithm(ts);
// bn_X4.setUseADTree(true);
// classifiers.add(bn_X4);

// BayesNet bn_X5 = new BayesNet();
//// -_0__wcbbn_00_D REMOVED
//// -_0__wcbbn_01_Q weka.classifiers.bayes.net.search.local.HillClimber
// HillClimber hc1 = new HillClimber();
// bn_X5.setSearchAlgorithm(hc1);
// bn_X5.setUseADTree(false);
// classifiers.add(bn_X5);
//
//// BayesNet bn_X6 = new BayesNet();
////// -_0__wcbbn_00_D REMOVE_PREV
////// -_0__wcbbn_01_Q weka.classifiers.bayes.net.search.local.LAGDHillClimber
//// LAGDHillClimber ldc0 = new LAGDHillClimber();
//// bn_X6.setSearchAlgorithm(ldc0);
//// bn_X6.setUseADTree(true);
//// classifiers.add(bn_X6);
//
//
//// BayesNet bn_X7 = new BayesNet();
////// -_0__wcbbn_00_D REMOVED
////// -_0__wcbbn_01_Q weka.classifiers.bayes.net.search.local.LAGDHillClimber
////
//// LAGDHillClimber ldc = new LAGDHillClimber();
//// bn_X7.setSearchAlgorithm(ldc);
//// bn_X7.setUseADTree(false);
//// classifiers.add(bn_X7);
//
// BayesNet bn_X8 = new BayesNet();
//// -_0__wcbbn_00_D REMOVE_PREV
//// -_0__wcbbn_01_Q weka.classifiers.bayes.net.search.local.TabuSearch
// TabuSearch ts1 = new TabuSearch();
// bn_X8.setSearchAlgorithm(ts1);
// bn_X8.setUseADTree(true);
// classifiers.add(bn_X8);
//
// BayesNet bn_X9 = new BayesNet();
//// -_0__wcbbn_00_D REMOVE_PREV
//// -_0__wcbbn_01_Q weka.classifiers.bayes.net.search.local.K2
// K2 k2 = new K2();
// bn_X9.setSearchAlgorithm(k2);
// bn_X9.setUseADTree(true);
// classifiers.add(bn_X9);
//
//// BayesNet bn_X10 = new BayesNet();
////// -_0__wcbbn_00_D REMOVED
////// -_0__wcbbn_01_Q weka.classifiers.bayes.net.search.local.LAGDHillClimber
//// LAGDHillClimber ldc1 = new LAGDHillClimber();
//// bn_X10.setSearchAlgorithm(ldc1);
//// bn_X10.setUseADTree(false);
//// classifiers.add(bn_X10);

// NaiveBayes nb = new NaiveBayes();
//// nb.setUseSupervisedDiscretization(true);
// classifiers.add(nb);

// NaiveBayes nb_X2 = new NaiveBayes();
//// -_0__wcbnb_00_K REMOVED -_0__wcbnb_01_D REMOVE_PREV
// nb_X2.setUseKernelEstimator(true);
// nb_X2.setUseSupervisedDiscretization(false);
// classifiers.add(nb_X2);
//
// NaiveBayes nb_X3 = new NaiveBayes();
//// -_0__wcbnb_00_K REMOVED -_0__wcbnb_01_D REMOVE_PREV
// nb_X3.setUseKernelEstimator(true);
// nb_X3.setUseSupervisedDiscretization(false);
// classifiers.add(nb_X3);
//
// NaiveBayes nb_X4 = new NaiveBayes();
//// -_0__wcbnb_00_K REMOVED -_0__wcbnb_01_D REMOVE_PREV
// nb_X4.setUseKernelEstimator(true);
// nb_X4.setUseSupervisedDiscretization(false);
// classifiers.add(nb_X4);
//
// NaiveBayes nb_X5 = new NaiveBayes();
//// -_0__wcbnb_00_K REMOVE_PREV -_0__wcbnb_01_D REMOVED
// nb_X5.setUseKernelEstimator(false);
// nb_X5.setUseSupervisedDiscretization(true);
// classifiers.add(nb_X5);
//
// NaiveBayes nb_X6 = new NaiveBayes();
//// -_0__wcbnb_00_K REMOVE_PREV -_0__wcbnb_01_D REMOVED
// nb_X6.setUseKernelEstimator(false);
// nb_X6.setUseSupervisedDiscretization(true);
// classifiers.add(nb_X6);
//
// NaiveBayes nb_X7 = new NaiveBayes();
//// -_0__wcbnb_00_K REMOVE_PREV -_0__wcbnb_01_D REMOVED
// nb_X7.setUseKernelEstimator(false);
// nb_X7.setUseSupervisedDiscretization(true);
// classifiers.add(nb_X7);
//
// NaiveBayes nb_X8 = new NaiveBayes();
//// -_0__wcbnb_00_K REMOVE_PREV -_0__wcbnb_01_D REMOVED
// nb_X8.setUseKernelEstimator(false);
// nb_X8.setUseSupervisedDiscretization(true);
// classifiers.add(nb_X8);
//
// NaiveBayes nb_X9 = new NaiveBayes();
//// -_0__wcbnb_00_K REMOVE_PREV -_0__wcbnb_01_D REMOVED
// nb_X9.setUseKernelEstimator(false);
// nb_X9.setUseSupervisedDiscretization(true);
// classifiers.add(nb_X9);
//
// NaiveBayes nb_X10 = new NaiveBayes();
//// -_0__wcbnb_00_K REMOVE_PREV -_0__wcbnb_01_D REMOVED
// nb_X10.setUseKernelEstimator(false);
// nb_X10.setUseSupervisedDiscretization(true);
// classifiers.add(nb_X10);

// SimpleLogistic sl = new SimpleLogistic();
//// sl.setUseCrossValidation(false);
//// sl.setWeightTrimBeta(0.2858669348731461);
// classifiers.add(sl);
//
// SimpleLogistic sl_X2 = new SimpleLogistic();
//// -_0__wcfsl_00_S REMOVE_PREV
//// -_0__wcfsl_01_W_HIDDEN 1 -_0__wcfsl_03_2_W 0.7308188691758736
//// -_0__wcfsl_04_A REMOVED
// sl_X2.setUseCrossValidation(true);
// sl_X2.setWeightTrimBeta(0.7308188691758736);
// sl_X2.setUseAIC(true);
// classifiers.add(sl_X2);
//
// SimpleLogistic sl_X3 = new SimpleLogistic();
//// -_0__wcfsl_00_S REMOVE_PREV
//// -_0__wcfsl_01_W_HIDDEN 1 -_0__wcfsl_03_2_W 0.28979247817934395
//// -_0__wcfsl_04_A REMOVE_PREV
// sl_X3.setUseCrossValidation(true);
// sl_X3.setWeightTrimBeta(0.28979247817934395);
// sl_X3.setUseAIC(false);
// classifiers.add(sl_X3);
//
// SimpleLogistic sl_X4 = new SimpleLogistic();
//// -_0__wcfsl_00_S REMOVE_PREV
//// -_0__wcfsl_01_W_HIDDEN 1 -_0__wcfsl_03_2_W 0.09571888870158973
//// -_0__wcfsl_04_A REMOVE_PREV
// sl_X4.setUseCrossValidation(true);
// sl_X4.setWeightTrimBeta(0.09571888870158973);
// sl_X4.setUseAIC(false);
// classifiers.add(sl_X4);

// SimpleLogistic sl_X5 = new SimpleLogistic();
//// -_0__wcfsl_00_S REMOVED
//// -_0__wcfsl_01_W_HIDDEN 1 -_0__wcfsl_03_2_W 0.14712520703073284
//// -_0__wcfsl_04_A REMOVED
// sl_X5.setUseCrossValidation(false);
// sl_X5.setWeightTrimBeta(0.14712520703073284);
// sl_X5.setUseAIC(true);
// classifiers.add(sl_X5);

// SimpleLogistic sl_X6 = new SimpleLogistic();
//// -_0__wcfsl_00_S REMOVED
//// -_0__wcfsl_01_W_HIDDEN 1 -_0__wcfsl_03_2_W 0.1656265331414748
//// -_0__wcfsl_04_A REMOVE_PREV
// sl_X6.setUseCrossValidation(false);
// sl_X6.setWeightTrimBeta(0.1656265331414748);
// sl_X6.setUseAIC(false);
// classifiers.add(sl_X6);
//
// SimpleLogistic sl_X7 = new SimpleLogistic();
//// -_0__wcfsl_00_S REMOVED
//// -_0__wcfsl_01_W_HIDDEN 1 -_0__wcfsl_03_2_W 0.005061617653999595
//// -_0__wcfsl_04_A REMOVE_PREV
// sl_X7.setUseCrossValidation(false);
// sl_X7.setWeightTrimBeta(0.005061617653999595);
// sl_X7.setUseAIC(false);
// classifiers.add(sl_X7);
//
// SimpleLogistic sl_X8 = new SimpleLogistic();
//// -_0__wcfsl_00_S REMOVED
//// -_0__wcfsl_01_W_HIDDEN 1 -_0__wcfsl_03_2_W 0.001752174957156738
//// -_0__wcfsl_04_A REMOVE_PREV
// sl_X8.setUseCrossValidation(false);
// sl_X8.setWeightTrimBeta(0.001752174957156738);
// sl_X8.setUseAIC(false);
// classifiers.add(sl_X8);
//
// SimpleLogistic sl_X9 = new SimpleLogistic();
//// -_0__wcfsl_00_S REMOVED
//// -_0__wcfsl_01_W_HIDDEN 1 -_0__wcfsl_03_2_W 0.15842012627261637
//// -_0__wcfsl_04_A REMOVE_PREV
// sl_X9.setUseCrossValidation(false);
// sl_X9.setWeightTrimBeta(0.15842012627261637);
// sl_X9.setUseAIC(false);
// classifiers.add(sl_X9);
//
// SimpleLogistic sl_X10 = new SimpleLogistic();
//// -_0__wcfsl_00_S REMOVED
//// -_0__wcfsl_01_W_HIDDEN 1 -_0__wcfsl_03_2_W 0.09116794321222899
//// -_0__wcfsl_04_A REMOVE_PREV
// sl_X10.setUseCrossValidation(false);
// sl_X10.setWeightTrimBeta(0.09116794321222899);
// sl_X10.setUseAIC(false);
// classifiers.add(sl_X10);

// Logistic logistic = new Logistic();
//// logistic.setRidge(9.955544709104409);
// classifiers.add(logistic);
//
// Logistic logistic_X2 = new Logistic();
// logistic_X2.setRidge(2.2804727375749456E-4);
// classifiers.add(logistic_X2);
//
// Logistic logistic_X3 = new Logistic();
// logistic_X3.setRidge(2.2804727375749456E-4);
// classifiers.add(logistic_X3);
//
// Logistic logistic_X4 = new Logistic();
// logistic_X4.setRidge(0.02786417144586695);
// classifiers.add(logistic_X4);
//
// Logistic logistic_X5 = new Logistic();
// logistic_X5.setRidge(2.2804727375749456E-4);
// classifiers.add(logistic_X5);
//
// Logistic logistic_X6 = new Logistic();
//// -_0__wcfl_00_R 3.3269160155362107
// logistic_X6.setRidge(3.3269160155362107);
// classifiers.add(logistic_X6);
//
// Logistic logistic_X7 = new Logistic();
// logistic_X7.setRidge(3.1483675693872493);
// classifiers.add(logistic_X7);
//
// Logistic logistic_X8 = new Logistic();
// logistic_X8.setRidge(4.119713591130133);
// classifiers.add(logistic_X8);

// Logistic logistic_X9 = new Logistic();
// logistic_X9.setRidge(7.611593557575053);
// classifiers.add(logistic_X9);

// Logistic logistic_X10 = new Logistic();
// logistic_X10.setRidge(4.282314924883062);
// classifiers.add(logistic_X10);
//
//
//
// J48 j48 = new J48();
//// j48.setCollapseTree(false);
//// j48.setBinarySplits(false);
//// j48.setUseMDLcorrection(true);
//// j48.setUseLaplace(true);
//// j48.setSubtreeRaising(false);
//// j48.setMinNumObj(1);
//// j48.setConfidenceFactor((float)0.1448342966182189);
// classifiers.add(j48);
//
// J48 j48_X2 = new J48();
//// -_0__wctj_00_O REMOVED
//// -_0__wctj_01_U REMOVE_PREV
//// -_0__wctj_02_B REMOVED
//// -_0__wctj_03_J REMOVE_PREV
//// -_0__wctj_04_A REMOVE_PREV
//// -_0__wctj_05_S REMOVE_PREV
//// -_0__wctj_06_INT_M 1
//// -_0__wctj_07_C 0.250112844141703
// j48_X2.setCollapseTree(false); // -_0__wctj_00_O REMOVED
// j48_X2.setUnpruned(false); // -_0__wctj_01_U REMOVE_PREV
// j48_X2.setBinarySplits(true); // -_0__wctj_02_B REMOVED
// j48_X2.setUseMDLcorrection(true); // -_0__wctj_03_J REMOVE_PREV
// j48_X2.setUseLaplace(true); // -_0__wctj_04_A REMOVE_PREV
// j48_X2.setSubtreeRaising(true); // -_0__wctj_05_S REMOVE_PREV
// j48_X2.setMinNumObj(1); // -_0__wctj_06_INT_M 1
// j48_X2.setConfidenceFactor((float) 0.250112844141703); // -_0__wctj_07_C
// 0.250112844141703
// classifiers.add(j48_X2);
//
// J48 j48_X3 = new J48();
//// -_0__wctj_00_O REMOVED
//// -_0__wctj_01_U REMOVE_PREV
//// -_0__wctj_02_B REMOVE_PREV
//// -_0__wctj_03_J REMOVE_PREV
//// -_0__wctj_04_A REMOVE_PREV
//// -_0__wctj_05_S REMOVE_PREV
//// -_0__wctj_06_INT_M 3
//// -_0__wctj_07_C 0.21257222026433878
// j48_X3.setCollapseTree(false); // -_0__wctj_00_O REMOVED
// j48_X3.setUnpruned(false); // -_0__wctj_01_U REMOVE_PREV
// j48_X3.setBinarySplits(false); // -_0__wctj_02_B REMOVE_PREV
// j48_X3.setUseMDLcorrection(true); // -_0__wctj_03_J REMOVE_PREV
// j48_X3.setUseLaplace(true); // -_0__wctj_04_A REMOVE_PREV
// j48_X3.setSubtreeRaising(true); // -_0__wctj_05_S REMOVE_PREV
// j48_X3.setMinNumObj(3); // -_0__wctj_06_INT_M 3
// j48_X3.setConfidenceFactor((float) 0.21257222026433878); // -_0__wctj_07_C
// 0.21257222026433878
// classifiers.add(j48_X3);

// J48 j48_X4 = new J48();
//// -_0__wctj_00_O REMOVED
//// -_0__wctj_01_U REMOVE_PREV
//// -_0__wctj_02_B REMOVE_PREV
//// -_0__wctj_03_J REMOVE_PREV
//// -_0__wctj_04_A REMOVE_PREV
//// -_0__wctj_05_S REMOVED
//// -_0__wctj_06_INT_M 4
//// -_0__wctj_07_C 0.3037311112514006
// j48_X4.setCollapseTree(false); // -_0__wctj_00_O REMOVED
// j48_X4.setUnpruned(false); // -_0__wctj_01_U REMOVE_PREV
// j48_X4.setBinarySplits(false); // -_0__wctj_02_B REMOVE_PREV
// j48_X4.setUseMDLcorrection(true); // -_0__wctj_03_J REMOVE_PREV
// j48_X4.setUseLaplace(true); // -_0__wctj_04_A REMOVE_PREV
// j48_X4.setSubtreeRaising(false); // -_0__wctj_05_S REMOVED
// j48_X4.setMinNumObj(4); // -_0__wctj_06_INT_M 4
// j48_X4.setConfidenceFactor((float) 0.3037311112514006); // -_0__wctj_07_C
// 0.3037311112514006
// classifiers.add(j48_X4);

// J48 j48_X5 = new J48();
//// -_0__wctj_00_O REMOVED
//// -_0__wctj_01_U REMOVE_PREV
//// -_0__wctj_02_B REMOVE_PREV
//// -_0__wctj_03_J REMOVE_PREV
//// -_0__wctj_04_A REMOVE_PREV
//// -_0__wctj_05_S REMOVED
//// -_0__wctj_06_INT_M 4
//// -_0__wctj_07_C 0.3037311112514006
// j48_X5.setCollapseTree(false); // -_0__wctj_00_O REMOVED
// j48_X5.setUnpruned(false); // -_0__wctj_01_U REMOVE_PREV
// j48_X5.setBinarySplits(false); // -_0__wctj_02_B REMOVE_PREV
// j48_X5.setUseMDLcorrection(true); // -_0__wctj_03_J REMOVE_PREV
// j48_X5.setUseLaplace(true); // -_0__wctj_04_A REMOVE_PREV
// j48_X5.setSubtreeRaising(false); // -_0__wctj_05_S REMOVED
// j48_X5.setMinNumObj(4); // -_0__wctj_06_INT_M 4
// j48_X5.setConfidenceFactor((float) 0.3037311112514006); // -_0__wctj_07_C
// 0.3037311112514006
// classifiers.add(j48_X5);
//
// J48 j48_X6 = new J48();
//// -_0__wctj_00_O REMOVED
//// -_0__wctj_01_U REMOVE_PREV
//// -_0__wctj_02_B REMOVED
//// -_0__wctj_03_J REMOVE_PREV
//// -_0__wctj_04_A REMOVE_PREV
//// -_0__wctj_05_S REMOVED
//// -_0__wctj_06_INT_M 4
//// -_0__wctj_07_C 0.6071306647539894
// j48_X6.setCollapseTree(false); // -_0__wctj_00_O REMOVED
// j48_X6.setUnpruned(false); // -_0__wctj_01_U REMOVE_PREV
// j48_X6.setBinarySplits(true); // -_0__wctj_02_B REMOVED
// j48_X6.setUseMDLcorrection(true); // -_0__wctj_03_J REMOVE_PREV
// j48_X6.setUseLaplace(true); // -_0__wctj_04_A REMOVE_PREV
// j48_X6.setSubtreeRaising(false); // -_0__wctj_05_S REMOVED
// j48_X6.setMinNumObj(4); // -_0__wctj_06_INT_M 4
// j48_X6.setConfidenceFactor((float) 0.6071306647539894); // -_0__wctj_07_C
// 0.6071306647539894
// classifiers.add(j48_X6);
//
// J48 j48_X7 = new J48();
//// -_0__wctj_00_O REMOVED
//// -_0__wctj_01_U REMOVE_PREV
//// -_0__wctj_02_B REMOVE_PREV
//// -_0__wctj_03_J REMOVE_PREV
//// -_0__wctj_04_A REMOVE_PREV
//// -_0__wctj_05_S REMOVE_PREV
//// -_0__wctj_06_INT_M 13
//// -_0__wctj_07_C 0.3170549061536089
// j48_X7.setCollapseTree(false); // -_0__wctj_00_O REMOVED
// j48_X7.setUnpruned(false); // -_0__wctj_01_U REMOVE_PREV
// j48_X7.setBinarySplits(false); // -_0__wctj_02_B REMOVE_PREV
// j48_X7.setUseMDLcorrection(true); // -_0__wctj_03_J REMOVE_PREV
// j48_X7.setUseLaplace(true); // -_0__wctj_04_A REMOVE_PREV
// j48_X7.setSubtreeRaising(true); // -_0__wctj_05_S REMOVE_PREV
// j48_X7.setMinNumObj(13); // -_0__wctj_06_INT_M 3
// j48_X7.setConfidenceFactor((float) 0.3170549061536089); // -_0__wctj_07_C
// 0.3170549061536089
// classifiers.add(j48_X7);
//
// J48 j48_X8 = new J48();
//// -_0__wctj_00_O REMOVE_PREV
//// -_0__wctj_01_U REMOVE_PREV
//// -_0__wctj_02_B REMOVED
//// -_0__wctj_03_J REMOVED
//// -_0__wctj_04_A REMOVE_PREV
//// -_0__wctj_05_S REMOVED
//// -_0__wctj_06_INT_M 3
//// -_0__wctj_07_C 0.20877965347112792
// j48_X8.setCollapseTree(true); // -_0__wctj_00_O REMOVE_PREV
// j48_X8.setUnpruned(false); // -_0__wctj_01_U REMOVE_PREV
// j48_X8.setBinarySplits(true); // -_0__wctj_02_B REMOVED
// j48_X8.setUseMDLcorrection(false); // -_0__wctj_03_J REMOVED
// j48_X8.setUseLaplace(true); // -_0__wctj_04_A REMOVE_PREV
// j48_X8.setSubtreeRaising(false); // -_0__wctj_05_S REMOVED
// j48_X8.setMinNumObj(3); // -_0__wctj_06_INT_M 3
// j48_X8.setConfidenceFactor((float) 0.20877965347112792); // -_0__wctj_07_C
// 0.20877965347112792
// classifiers.add(j48_X8);
//
// J48 j48_X9 = new J48();
//// -_0__wctj_00_O REMOVED
//// -_0__wctj_01_U REMOVE_PREV
//// -_0__wctj_02_B REMOVE_PREV
//// -_0__wctj_03_J REMOVED
//// -_0__wctj_04_A REMOVED
//// -_0__wctj_05_S REMOVE_PREV
//// -_0__wctj_06_INT_M 1
//// -_0__wctj_07_C 0.1146260555786317
// j48_X9.setCollapseTree(false); // -_0__wctj_00_O REMOVED
// j48_X9.setUnpruned(false); // -_0__wctj_01_U REMOVE_PREV
// j48_X9.setBinarySplits(false); // -_0__wctj_02_B REMOVE_PREV
// j48_X9.setUseMDLcorrection(false); // -_0__wctj_03_J REMOVED
// j48_X9.setUseLaplace(false); // -_0__wctj_04_A REMOVED
// j48_X9.setSubtreeRaising(true); // -_0__wctj_05_S REMOVE_PREV
// j48_X9.setMinNumObj(1); // -_0__wctj_06_INT_M 1
// j48_X9.setConfidenceFactor((float) 0.1146260555786317); // -_0__wctj_07_C
// 0.1146260555786317
// classifiers.add(j48_X9);
//
// J48 j48_X10 = new J48();
//// -_0__wctj_00_O REMOVED
//// -_0__wctj_01_U REMOVE_PREV
//// -_0__wctj_02_B REMOVE_PREV
//// -_0__wctj_03_J REMOVED
//// -_0__wctj_04_A REMOVE_PREV
//// -_0__wctj_05_S REMOVED
//// -_0__wctj_06_INT_M 3
//// -_0__wctj_07_C 0.15086496613370493
// j48_X10.setCollapseTree(false); // -_0__wctj_00_O REMOVED
// j48_X10.setUnpruned(false); // -_0__wctj_01_U REMOVE_PREV
// j48_X10.setBinarySplits(false); // -_0__wctj_02_B REMOVE_PREV
// j48_X10.setUseMDLcorrection(false); // -_0__wctj_03_J REMOVED
// j48_X10.setUseLaplace(true); // -_0__wctj_04_A REMOVE_PREV
// j48_X10.setSubtreeRaising(false); // -_0__wctj_05_S REMOVED
// j48_X10.setMinNumObj(3); // -_0__wctj_06_INT_M 3
// j48_X10.setConfidenceFactor((float) 0.15086496613370493); //-_0__wctj_07_C
// 0.15086496613370493
// classifiers.add(j48_X10);

// LMT lmt = new LMT();
//// lmt.setFastRegression(false);
//// lmt.setErrorOnProbabilities(true);
//// lmt.setMinNumInstances(1);
//// lmt.setWeightTrimBeta(0.2573514645010736);
//// lmt.setUseAIC(true);
// classifiers.add(lmt);
//
// LMT lmt_X2 = new LMT();
//// -_0__wctlmt_00_B REMOVE_PREV
//// -_0__wctlmt_01_R REMOVED
//// -_0__wctlmt_02_C REMOVED
//// -_0__wctlmt_03_P REMOVED
//// -_0__wctlmt_04_INT_M 64
//// -_0__wctlmt_05_W_HIDDEN 1
//// -_0__wctlmt_07_2_W 0.03069281186258721
//// -_0__wctlmt_08_A REMOVED
// lmt_X2.setConvertNominal(false); // -_0__wctlmt_00_B REMOVE_PREV
// lmt_X2.setSplitOnResiduals(true); // -_0__wctlmt_01_R REMOVED
// lmt_X2.setFastRegression(false); // -_0__wctlmt_02_C REMOVED
// lmt_X2.setErrorOnProbabilities(true); // -_0__wctlmt_03_P REMOVED
// lmt_X2.setMinNumInstances(64); // -_0__wctlmt_04_INT_M 64
// lmt_X2.setWeightTrimBeta(0.03069281186258721); // -_0__wctlmt_07_2_W
// 0.03069281186258721
// lmt_X2.setUseAIC(true); // -_0__wctlmt_08_A REMOVED
// classifiers.add(lmt_X2);
//
// LMT lmt_X3 = new LMT();
//// -_0__wctlmt_00_B REMOVE_PREV
//// -_0__wctlmt_01_R REMOVE_PREV
//// -_0__wctlmt_02_C REMOVED
//// -_0__wctlmt_03_P REMOVE_PREV
//// -_0__wctlmt_04_INT_M 8
//// -_0__wctlmt_05_W_HIDDEN 1
//// -_0__wctlmt_07_2_W 0.8543894103479136
//// -_0__wctlmt_08_A REMOVE_PREV
// lmt_X3.setConvertNominal(false); // -_0__wctlmt_00_B REMOVE_PREV
// lmt_X3.setSplitOnResiduals(false); // -_0__wctlmt_01_R REMOVE_PREV
// lmt_X3.setFastRegression(false); // -_0__wctlmt_02_C REMOVED
// lmt_X3.setErrorOnProbabilities(false); // -_0__wctlmt_03_P REMOVE_PREV
// lmt_X3.setMinNumInstances(8); // -_0__wctlmt_04_INT_M 8
// lmt_X3.setWeightTrimBeta(0.8543894103479136); // -_0__wctlmt_07_2_W
// 0.8543894103479136
// lmt_X3.setUseAIC(false); // -_0__wctlmt_08_A REMOVE_PREV
// classifiers.add(lmt_X3);
//
// LMT lmt_X4 = new LMT();
//// -_0__wctlmt_00_B REMOVED
//// -_0__wctlmt_01_R REMOVE_PREV
//// -_0__wctlmt_02_C REMOVED
//// -_0__wctlmt_03_P REMOVE_PREV
//// -_0__wctlmt_04_INT_M 31
//// -_0__wctlmt_05_W_HIDDEN 0
//// -_0__wctlmt_06_1_W 0
//// -_0__wctlmt_08_A REMOVE_PREV
// lmt_X4.setConvertNominal(true); // -_0__wctlmt_00_B REMOVED
// lmt_X4.setSplitOnResiduals(false); // -_0__wctlmt_01_R REMOVE_PREV
// lmt_X4.setFastRegression(false); // -_0__wctlmt_02_C REMOVED
// lmt_X4.setErrorOnProbabilities(false); // -_0__wctlmt_03_P REMOVE_PREV
// lmt_X4.setMinNumInstances(31); // -_0__wctlmt_04_INT_M 31
// lmt_X4.setWeightTrimBeta(0); // -_0__wctlmt_06_1_W 0
// lmt_X4.setUseAIC(false); // -_0__wctlmt_08_A REMOVE_PREV
// classifiers.add(lmt_X4);
//
// LMT lmt_X5 = new LMT();
//// -_0__wctlmt_00_B REMOVE_PREV
//// -_0__wctlmt_01_R REMOVE_PREV
//// -_0__wctlmt_02_C REMOVED
//// -_0__wctlmt_03_P REMOVE_PREV
//// -_0__wctlmt_04_INT_M 9
//// -_0__wctlmt_05_W_HIDDEN 0
//// -_0__wctlmt_06_1_W 0
//// -_0__wctlmt_08_A REMOVED
// lmt_X5.setConvertNominal(false); // -_0__wctlmt_00_B REMOVE_PREV
// lmt_X5.setSplitOnResiduals(false); // -_0__wctlmt_01_R REMOVE_PREV
// lmt_X5.setFastRegression(false); // -_0__wctlmt_02_C REMOVED
// lmt_X5.setErrorOnProbabilities(false); // -_0__wctlmt_03_P REMOVE_PREV
// lmt_X5.setMinNumInstances(9); // -_0__wctlmt_04_INT_M 9
// lmt_X5.setWeightTrimBeta(0); // -_0__wctlmt_06_1_W 0
// lmt_X5.setUseAIC(true); // -_0__wctlmt_08_A REMOVED
// classifiers.add(lmt_X5);
//
// LMT lmt_X6 = new LMT();
//// -_0__wctlmt_00_B REMOVED
//// -_0__wctlmt_01_R REMOVE_PREV
//// -_0__wctlmt_02_C REMOVED
//// -_0__wctlmt_03_P REMOVE_PREV
//// -_0__wctlmt_04_INT_M 31
//// -_0__wctlmt_05_W_HIDDEN 0
//// -_0__wctlmt_06_1_W 0
//// -_0__wctlmt_08_A REMOVE_PREV
// lmt_X6.setConvertNominal(true); // -_0__wctlmt_00_B REMOVED
// lmt_X6.setSplitOnResiduals(false); // -_0__wctlmt_01_R REMOVE_PREV
// lmt_X6.setFastRegression(false); // -_0__wctlmt_02_C REMOVED
// lmt_X6.setErrorOnProbabilities(false); // -_0__wctlmt_03_P REMOVE_PREV
// lmt_X6.setMinNumInstances(31); // -_0__wctlmt_04_INT_M 31
// lmt_X6.setWeightTrimBeta(0); // -_0__wctlmt_06_1_W 0
// lmt_X6.setUseAIC(false); // -_0__wctlmt_08_A REMOVE_PREV
// classifiers.add(lmt_X6);

// LMT lmt_X7 = new LMT();
//// -_0__wctlmt_00_B REMOVED
//// -_0__wctlmt_01_R REMOVED
//// -_0__wctlmt_02_C REMOVE_PREV
//// -_0__wctlmt_03_P REMOVE_PREV
//// -_0__wctlmt_04_INT_M 18
//// -_0__wctlmt_05_W_HIDDEN 1
//// -_0__wctlmt_07_2_W 0.27512312096045455
//// -_0__wctlmt_08_A REMOVED
// lmt_X7.setConvertNominal(true); // -_0__wctlmt_00_B REMOVED
// lmt_X7.setSplitOnResiduals(true); // -_0__wctlmt_01_R REMOVED
// lmt_X7.setFastRegression(true); // -_0__wctlmt_02_C REMOVE_PREV
// lmt_X7.setErrorOnProbabilities(false); // -_0__wctlmt_03_P REMOVE_PREV
// lmt_X7.setMinNumInstances(18); // -_0__wctlmt_04_INT_M 18
// lmt_X7.setWeightTrimBeta(0.27512312096045455); // -_0__wctlmt_07_2_W
// 0.27512312096045455
// lmt_X7.setUseAIC(true); // -_0__wctlmt_08_A REMOVED
// classifiers.add(lmt_X7);

// LMT lmt_X8 = new LMT();
//// -_0__wctlmt_00_B REMOVED
//// -_0__wctlmt_01_R REMOVED
//// -_0__wctlmt_02_C REMOVED
//// -_0__wctlmt_03_P REMOVED
//// -_0__wctlmt_04_INT_M 7
//// -_0__wctlmt_05_W_HIDDEN 1
//// -_0__wctlmt_07_2_W 0.1096621932256876
//// -_0__wctlmt_08_A REMOVED
// lmt_X8.setConvertNominal(true); // -_0__wctlmt_00_B REMOVED
// lmt_X8.setSplitOnResiduals(true); // -_0__wctlmt_01_R REMOVED
// lmt_X8.setFastRegression(false); // -_0__wctlmt_02_C REMOVED
// lmt_X8.setErrorOnProbabilities(true); // -_0__wctlmt_03_P REMOVED
// lmt_X8.setMinNumInstances(7); // -_0__wctlmt_04_INT_M 7
// lmt_X8.setWeightTrimBeta(0.1096621932256876); // -_0__wctlmt_07_2_W
// 0.1096621932256876
// lmt_X8.setUseAIC(true); // -_0__wctlmt_08_A REMOVED
// classifiers.add(lmt_X8);
//
// LMT lmt_X9 = new LMT();
//// -_0__wctlmt_00_B REMOVED
//// -_0__wctlmt_01_R REMOVE_PREV
//// -_0__wctlmt_02_C REMOVE_PREV
//// -_0__wctlmt_03_P REMOVED
//// -_0__wctlmt_04_INT_M 1
//// -_0__wctlmt_05_W_HIDDEN 0
//// -_0__wctlmt_06_1_W 0
// lmt_X9.setFastRegression(true); // -_0__wctlmt_02_C REMOVE_PREV
//// -_0__wctlmt_08_A REMOVED
// lmt_X9.setConvertNominal(true); // -_0__wctlmt_00_B REMOVED
// lmt_X9.setSplitOnResiduals(false); // -_0__wctlmt_01_R REMOVE_PREV
// lmt_X9.setErrorOnProbabilities(true); // -_0__wctlmt_03_P REMOVED
// lmt_X9.setMinNumInstances(1); // -_0__wctlmt_04_INT_M 1
// lmt_X9.setWeightTrimBeta(0); // -_0__wctlmt_06_1_W 0
// lmt_X9.setUseAIC(true); // -_0__wctlmt_08_A REMOVED
// classifiers.add(lmt_X9);
//
// LMT lmt_X10 = new LMT();
//// -_0__wctlmt_00_B REMOVE_PREV
//// -_0__wctlmt_01_R REMOVED
//// -_0__wctlmt_02_C REMOVE_PREV
//// -_0__wctlmt_03_P REMOVE_PREV
//// -_0__wctlmt_04_INT_M 32
//// -_0__wctlmt_05_W_HIDDEN 0
//// -_0__wctlmt_06_1_W 0
//// -_0__wctlmt_08_A REMOVE_PREV
// lmt_X10.setConvertNominal(false); // -_0__wctlmt_00_B REMOVE_PREV
// lmt_X10.setSplitOnResiduals(true); // -_0__wctlmt_01_R REMOVED
// lmt_X10.setFastRegression(true); // -_0__wctlmt_02_C REMOVE_PREV
// lmt_X10.setErrorOnProbabilities(false); // -_0__wctlmt_03_P REMOVE_PREV
// lmt_X10.setMinNumInstances(32); // -_0__wctlmt_04_INT_M 32
// lmt_X10.setWeightTrimBeta(0); // -_0__wctlmt_06_1_W 0
// lmt_X10.setUseAIC(false); // -_0__wctlmt_08_A REMOVE_PREV
// classifiers.add(lmt_X10);
//
//
//
// RandomForest rf = new RandomForest();
//// rf.setNumTrees(123);
//// rf.setNumFeatures(2);
//// rf.setMaxDepth(18);
// classifiers.add(rf);
//
// RandomForest rf_X2 = new RandomForest();
//// -_0__wctrf_00_INT_I 79
//// -_0__wctrf_01_features_HIDDEN 1
//// -_0__wctrf_03_2_INT_K 1
//// -_0__wctrf_04_depth_HIDDEN 0
//// -_0__wctrf_05_1_INT_depth 0
// rf_X2.setNumTrees(79); // -_0__wctrf_00_INT_I 79
// rf_X2.setNumFeatures(1); // -_0__wctrf_03_2_INT_K 1
// rf_X2.setMaxDepth(0); // -_0__wctrf_05_1_INT_depth 0
// classifiers.add(rf_X2);
//
// RandomForest rf_X3 = new RandomForest();
//// -_0__wctrf_00_INT_I 51
//// -_0__wctrf_01_features_HIDDEN 1
//// -_0__wctrf_03_2_INT_K 2
//// -_0__wctrf_04_depth_HIDDEN 1
//// -_0__wctrf_06_2_INT_depth 16
// rf_X3.setNumTrees(51); // -_0__wctrf_00_INT_I 51
// rf_X3.setNumFeatures(2); // -_0__wctrf_03_2_INT_K 2
// rf_X3.setMaxDepth(16); // -_0__wctrf_05_1_INT_depth 16
// classifiers.add(rf_X3);

// RandomForest rf_X4 = new RandomForest();
//// -_0__wctrf_00_INT_I 205
//// -_0__wctrf_01_features_HIDDEN 1
//// -_0__wctrf_03_2_INT_K 4
//// -_0__wctrf_04_depth_HIDDEN 1
//// -_0__wctrf_06_2_INT_depth 14
// rf_X4.setNumTrees(205); // -_0__wctrf_00_INT_I 205
// rf_X4.setNumFeatures(4); // -_0__wctrf_03_2_INT_K 4
// rf_X4.setMaxDepth(14); // -_0__wctrf_05_1_INT_depth 14
// classifiers.add(rf_X4);

// RandomForest rf_X5 = new RandomForest();
//// -_0__wctrf_00_INT_I 28
//// -_0__wctrf_01_features_HIDDEN 1
//// -_0__wctrf_03_2_INT_K 5
//// -_0__wctrf_04_depth_HIDDEN 1
//// -_0__wctrf_06_2_INT_depth 12
// rf_X5.setNumTrees(28); // -_0__wctrf_00_INT_I 28
// rf_X5.setNumFeatures(5); // -_0__wctrf_03_2_INT_K 5
// rf_X5.setMaxDepth(12); // -_0__wctrf_05_1_INT_depth 12
// classifiers.add(rf_X5);
//
// RandomForest rf_X6 = new RandomForest();
//// -_0__wctrf_00_INT_I 62
//// -_0__wctrf_01_features_HIDDEN 1
//// -_0__wctrf_03_2_INT_K 4
//// -_0__wctrf_04_depth_HIDDEN 0
//// -_0__wctrf_05_1_INT_depth 0
// rf_X6.setNumTrees(62); // -_0__wctrf_00_INT_I 62
// rf_X6.setNumFeatures(4); // -_0__wctrf_03_2_INT_K 4
// rf_X6.setMaxDepth(0); // -_0__wctrf_05_1_INT_depth 0
// classifiers.add(rf_X6);
//
// RandomForest rf_X7 = new RandomForest();
//// -_0__wctrf_00_INT_I 141
//// -_0__wctrf_01_features_HIDDEN 1
//// -_0__wctrf_03_2_INT_K 2
//// -_0__wctrf_04_depth_HIDDEN 0
//// -_0__wctrf_05_1_INT_depth 0
// rf_X7.setNumTrees(141); // -_0__wctrf_00_INT_I 141
// rf_X7.setNumFeatures(2); // -_0__wctrf_03_2_INT_K 2
// rf_X7.setMaxDepth(0); // -_0__wctrf_05_1_INT_depth 0
// classifiers.add(rf_X7);
//
// RandomForest rf_X8 = new RandomForest();
//// -_0__wctrf_00_INT_I 120
//// -_0__wctrf_01_features_HIDDEN 1
//// -_0__wctrf_03_2_INT_K 1
//// -_0__wctrf_04_depth_HIDDEN 0
//// -_0__wctrf_05_1_INT_depth 0
// rf_X8.setNumTrees(120); // -_0__wctrf_00_INT_I 120
// rf_X8.setNumFeatures(1); // -_0__wctrf_03_2_INT_K 1
// rf_X8.setMaxDepth(0); // -_0__wctrf_05_1_INT_depth 0
// classifiers.add(rf_X8);
//
// RandomForest rf_X9 = new RandomForest();
//// -_0__wctrf_00_INT_I 91
//// -_0__wctrf_01_features_HIDDEN 1
//// -_0__wctrf_03_2_INT_K 4
//// -_0__wctrf_04_depth_HIDDEN 1
//// -_0__wctrf_06_2_INT_depth 19
// rf_X9.setNumTrees(91); // -_0__wctrf_00_INT_I 91
// rf_X9.setNumFeatures(4); // -_0__wctrf_03_2_INT_K 4
// rf_X9.setMaxDepth(19); // -_0__wctrf_05_1_INT_depth 19
// classifiers.add(rf_X9);
//
// RandomForest rf_X10 = new RandomForest();
//// -_0__wctrf_00_INT_I 67
//// -_0__wctrf_01_features_HIDDEN 1
//// -_0__wctrf_03_2_INT_K 2
//// -_0__wctrf_04_depth_HIDDEN 0
//// -_0__wctrf_05_1_INT_depth 0
// rf_X10.setNumTrees(67); // -_0__wctrf_00_INT_I 67
// rf_X10.setNumFeatures(2); // -_0__wctrf_03_2_INT_K 2
// rf_X10.setMaxDepth(0); // -_0__wctrf_05_1_INT_depth 0
// classifiers.add(rf_X10);

// rf.setBatchSize("100");
// rf.setBreakTiesRandomly(false);
// rf.setDoNotCheckCapabilities(false);
// rf.setDontCalculateOutOfBagError(false);
// rf.setNumDecimalPlaces(2);
// rf.setNumExecutionSlots(1);
// rf.setPrintTrees(false);
// rf.setSeed(1);

// // generate errors when I was trying to use Experiment Constructor
// Bagging bagging = new Bagging();
// classifiers.add(bagging);
//
// // generate errors when I was trying to use Experiment Constructor
// ClassificationViaRegression cvr = new ClassificationViaRegression();
// classifiers.add(cvr);
//
// // generate errors when I was trying to use Experiment Constructor
// LogitBoost lb = new LogitBoost();
// classifiers.add(lb);
//
// // generate errors when I was trying to use Experiment Constructor
// MultiClassClassifier mcc = new MultiClassClassifier();
// classifiers.add(mcc);
//
// // generate errors when I was trying to use Experiment Constructor,
// // but it is ok to use "save as definition" to batch file,
// // what is behind?
// RandomSubSpace rsc = new RandomSubSpace();
// classifiers.add(rsc);
//
// // this one is not in the list I can pick
// NaiveBayesUpdateable nbu = new NaiveBayesUpdateable();
// classifiers.add(nbu);

////// =====>the following part is used to get the performances
////// =====>of classifiers with optimized hyper parameters
//
// BayesNet bn_X4 = new BayesNet();
//// -_0__wcbbn_00_D REMOVE_PREV
//// -_0__wcbbn_01_Q weka.classifiers.bayes.net.search.local.TabuSearch
// TabuSearch ts = new TabuSearch();
// bn_X4.setSearchAlgorithm(ts);
// bn_X4.setUseADTree(true);
// classifiers.add(bn_X4);
//
// NaiveBayes nb = new NaiveBayes();
//// nb.setUseSupervisedDiscretization(true);
// classifiers.add(nb);
//
// SimpleLogistic sl_X5 = new SimpleLogistic();
//// -_0__wcfsl_00_S REMOVED
//// -_0__wcfsl_01_W_HIDDEN 1 -_0__wcfsl_03_2_W 0.14712520703073284
//// -_0__wcfsl_04_A REMOVED
// sl_X5.setUseCrossValidation(false);
// sl_X5.setWeightTrimBeta(0.14712520703073284);
// sl_X5.setUseAIC(true);
// classifiers.add(sl_X5);
//
// Logistic logistic_X9 = new Logistic();
// logistic_X9.setRidge(7.611593557575053);
// classifiers.add(logistic_X9);
//
// J48 j48_X4 = new J48();
//// -_0__wctj_00_O REMOVED
//// -_0__wctj_01_U REMOVE_PREV
//// -_0__wctj_02_B REMOVE_PREV
//// -_0__wctj_03_J REMOVE_PREV
//// -_0__wctj_04_A REMOVE_PREV
//// -_0__wctj_05_S REMOVED
//// -_0__wctj_06_INT_M 4
//// -_0__wctj_07_C 0.3037311112514006
// j48_X4.setCollapseTree(false); // -_0__wctj_00_O REMOVED
// j48_X4.setUnpruned(false); // -_0__wctj_01_U REMOVE_PREV
// j48_X4.setBinarySplits(false); // -_0__wctj_02_B REMOVE_PREV
// j48_X4.setUseMDLcorrection(true); // -_0__wctj_03_J REMOVE_PREV
// j48_X4.setUseLaplace(true); // -_0__wctj_04_A REMOVE_PREV
// j48_X4.setSubtreeRaising(false); // -_0__wctj_05_S REMOVED
// j48_X4.setMinNumObj(4); // -_0__wctj_06_INT_M 4
// j48_X4.setConfidenceFactor((float) 0.3037311112514006); // -_0__wctj_07_C
////// 0.3037311112514006
// classifiers.add(j48_X4);
//
// LMT lmt_X7 = new LMT();
//// -_0__wctlmt_00_B REMOVED
//// -_0__wctlmt_01_R REMOVED
//// -_0__wctlmt_02_C REMOVE_PREV
//// -_0__wctlmt_03_P REMOVE_PREV
//// -_0__wctlmt_04_INT_M 18
//// -_0__wctlmt_05_W_HIDDEN 1
//// -_0__wctlmt_07_2_W 0.27512312096045455
//// -_0__wctlmt_08_A REMOVED
// lmt_X7.setConvertNominal(true); // -_0__wctlmt_00_B REMOVED
// lmt_X7.setSplitOnResiduals(true); // -_0__wctlmt_01_R REMOVED
// lmt_X7.setFastRegression(true); // -_0__wctlmt_02_C REMOVE_PREV
// lmt_X7.setErrorOnProbabilities(false); // -_0__wctlmt_03_P REMOVE_PREV
// lmt_X7.setMinNumInstances(18); // -_0__wctlmt_04_INT_M 18
// lmt_X7.setWeightTrimBeta(0.27512312096045455); // -_0__wctlmt_07_2_W
////// 0.27512312096045455
// lmt_X7.setUseAIC(true); // -_0__wctlmt_08_A REMOVED
// classifiers.add(lmt_X7);
//
//
// RandomForest rf_X4 = new RandomForest();
//// -_0__wctrf_00_INT_I 205
//// -_0__wctrf_01_features_HIDDEN 1
//// -_0__wctrf_03_2_INT_K 4
//// -_0__wctrf_04_depth_HIDDEN 1
//// -_0__wctrf_06_2_INT_depth 14
// rf_X4.setNumTrees(205); // -_0__wctrf_00_INT_I 205
// rf_X4.setNumFeatures(4); // -_0__wctrf_03_2_INT_K 4
// rf_X4.setMaxDepth(14); // -_0__wctrf_05_1_INT_depth 14
// classifiers.add(rf_X4);

//
//// this method is used to generate the combinations of the different hyper
//// parameters
// public static void print_HyperPara_Combinations(String Classifier_Name) {
// // read from a file for the possible combinations
// // the input file has to be in a certain format:
// // each section will be sperated by blank lines
//
// try {
//
// // main_path + "/" + "ibk" + "_input.txt"
// File f = new File(main_path + "/" + Classifier_Name + "_input.txt");
// BufferedReader bufRead = new BufferedReader(new InputStreamReader(new
// FileInputStream(f)));
//
// // assume we only need to take care of three types of combinations
// // otherwise, I need to change the a little bit here;
// LinkedList<String> part_1 = new LinkedList<String>();
// LinkedList<String> part_2 = new LinkedList<String>();
// LinkedList<String> part_3 = new LinkedList<String>();
// int part_index = 1;
//
// // A very interesting discussion about why the original codes does't
// // work (although it did work before and I am not sure the reason)
// //
// http://stackoverflow.com/questions/24934777/null-pointer-exception-when-reading-textfile
//
// // String one_Line = "";
// // while ((one_Line = bufRead.readLine().trim()) != null) {
//
// for (String one_Line; (one_Line = bufRead.readLine()) != null;) {
// if (!one_Line.equals("")) {
// if (part_index == 1) {
// part_1.add(one_Line);
// } else if (part_index == 2) {
// part_2.add(one_Line);
// } else if (part_index == 3) {
// part_3.add(one_Line);
// }
// } else {
// part_index++;
// }
// }
//
// bufRead.close();
//
// PrintWriter writer = new PrintWriter(main_path + "/" + Classifier_Name +
// "_Output.txt");
//
// for (int j = 0; j < part_1.size(); j++) {
// for (int m = 0; m < part_2.size(); m++) {
//
// if (part_3.isEmpty()) {
//
// System.out.println(part_1.get(j));
// System.out.println(part_2.get(m));
// System.out.println();
//
// writer.println(part_1.get(j));
// writer.println(part_2.get(m));
// writer.println();
//
// }
//
// else if (!part_3.isEmpty()) {
// for (int n = 0; n < part_3.size(); n++) {
//
// System.out.println(part_1.get(j));
// System.out.println(part_2.get(m));
// System.out.println(part_3.get(n));
// System.out.println();
//
// writer.println(part_1.get(j));
// writer.println(part_2.get(m));
// writer.println(part_3.get(n));
// writer.println();
// }
// }
// }
// }
//
// writer.close();
//
// } catch (IOException ioe) {
// System.out.println("Something is wrong with reading file! " + "Can't find the
// appropriate file!");
// System.out.println(ioe.toString());
// }
//
// // write by the combinations to a local file so that we can use
// // it by just copying and pasting into codes individually
// }
//
