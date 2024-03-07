from feature_extractor import FeatureExtractor

class SimpleFeatureExtractor(FeatureExtractor):
    @property
    def output_dim(self):
        return 128  # Example fixed output dimension

    def extract_features(self, data):
        print("Extracting features from data.")
        # Implement feature extraction logic here
        return "features"  # Placeholder return value