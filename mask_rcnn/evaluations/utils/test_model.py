import os
from mrcnn.config import Config
from mrcnn import model as modellib
from utils.custom_dataset import DurianLoculeDataset


def initialize_test_model(model_name: str, backbone: str):
    class LoculesConfig(Config):
        NAME = "locules"
        BACKBONE = backbone
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 2 + 1
        STEPS_PER_EPOCH = 100
        DETECTION_MIN_CONFIDENCE = 0.9
        IMAGE_RESIZE_MODE = "none"
        IMAGE_MIN_DIM = 640
        IMAGE_MAX_DIM = 640
        LEARNING_RATE = 0.001
        LEARNING_MOMENTUM = 0.9
        WEIGHT_DECAY = 0.0001

    config = LoculesConfig()
    config.display()

    ROOT_DIR = os.path.abspath("./")
    HOME = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    TEST_DATASET_JSON = "./dataset/test/_annotations.coco.json"
    TEST_DATASET_DIR = "./dataset/test/"
    TRAINED_MODEL_WEIGHTS_PATH = f"{HOME}/{model_name}"

    dataset_test = DurianLoculeDataset()
    dataset_test.load_data(TEST_DATASET_JSON, TEST_DATASET_DIR)
    dataset_test.prepare()

    test_config = LoculesConfig()
    test_model = modellib.MaskRCNN(
        mode="inference", config=test_config, model_dir=MODEL_DIR
    )
    test_model.load_weights(TRAINED_MODEL_WEIGHTS_PATH, by_name=True)

    return dataset_test, test_model, test_config
