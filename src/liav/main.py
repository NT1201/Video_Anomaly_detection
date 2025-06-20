# main.py
from feature_dataset import VideoFeatureDataset

if __name__ == '__main__':
    # training_dataset = VideoFeatureDataset("project/data/Training")
    testing_dataset = VideoFeatureDataset("C:\IOT_new\data\Testing")

    # print(f"Loaded {len(training_dataset)} training videos")
    print(f"Loaded {len(testing_dataset)} testing videos")