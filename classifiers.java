import java.util.HashMap;
import java.util.Map;

public class NaiveBayesClassifier {
    private Map<String, Integer> classCounts = new HashMap<>();
    private Map<String, Map<String, Integer>> featureCounts = new HashMap<>();
    private Map<String, Integer> featureTotalCounts = new HashMap<>();
    private int totalInstances = 0;

    public void train(String[] features, String label) {
        if (features == null || label == null) return;

        classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        totalInstances++;

        Map<String, Integer> featureMap = featureCounts.computeIfAbsent(label, k -> new HashMap<>());
        for (String feature : features) {
            featureMap.put(feature, featureMap.getOrDefault(feature, 0) + 1);
            featureTotalCounts.put(feature, featureTotalCounts.getOrDefault(feature, 0) + 1);
        }
    }

    public String predict(String[] features) {
        if (features == null) return null;

        double maxProbability = Double.NEGATIVE_INFINITY;
        String bestLabel = null;

        for (String label : classCounts.keySet()) {
            double labelProbability = Math.log((double) classCounts.get(label) / totalInstances);

            for (String feature : features) {
                double featureProbability = (featureCounts.getOrDefault(label, new HashMap<>()).getOrDefault(feature, 0) + 1) 
                                            / (double) (featureTotalCounts.getOrDefault(feature, 0) + featureCounts.getOrDefault(label, new HashMap<>()).size());
                labelProbability += Math.log(featureProbability);
            }

            if (labelProbability > maxProbability) {
                maxProbability = labelProbability;
                bestLabel = label;
            }
        }

        return bestLabel;
    }

    public static void main(String[] args) {
        NaiveBayesClassifier nb = new NaiveBayesClassifier();

        nb.train(new String[]{"sunny", "hot", "high", "false"}, "no");
        nb.train(new String[]{"sunny", "hot", "high", "true"}, "no");
        nb.train(new String[]{"overcast", "hot", "high", "false"}, "yes");
        nb.train(new String[]{"rainy", "mild", "high", "false"}, "yes");
        nb.train(new String[]{"rainy", "cool", "normal", "false"}, "yes");
        nb.train(new String[]{"rainy", "cool", "normal", "true"}, "no");
        nb.train(new String[]{"overcast", "cool", "normal", "true"}, "yes");
        nb.train(new String[]{"sunny", "mild", "high", "false"}, "no");
        nb.train(new String[]{"sunny", "cool", "normal", "false"}, "yes");
        nb.train(new String[]{"rainy", "mild", "normal", "false"}, "yes");
        nb.train(new String[]{"sunny", "mild", "normal", "true"}, "yes");
        nb.train(new String[]{"overcast", "mild", "high", "true"}, "yes");
        nb.train(new String[]{"overcast", "hot", "normal", "false"}, "yes");
        nb.train(new String[]{"rainy", "mild", "high", "true"}, "no");

        String[] testFeatures = {"sunny", "cool", "high", "true"};
        String prediction = nb.predict(testFeatures);
        System.out.println("Prediction: " + prediction);
    }
}