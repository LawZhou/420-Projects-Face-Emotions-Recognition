from emotionsClassifier import EmotionsClassifier
from evaluate import Evaluator
from featureExtraction import TrainKFoldModels, CountWrongPredictions
from knn import KNN
from preprocessCK import PreprocessCK
from helpers import saveModel, saveKnnModel, saveNumpys

class RunTraining:
    def __init__(self, dir):
        self.preprocess_ck = PreprocessCK(dir)
        self.k_fold_trainer = TrainKFoldModels()
        self.incorrects_counter = CountWrongPredictions()
        self.knn = KNN()
        self.emotion_classifier = EmotionsClassifier()

    def run_training(self):
        train_imgs, train_labels, val_imgs, val_labels = self.preprocess_ck.run_preprocess()
        all_K_fold_models = self.k_fold_trainer.train_K_Fold_models(train_imgs, train_labels)
        train_imgs, incorrect_labels = self.incorrects_counter.count_wrong_prediction_times(train_imgs, train_labels,
                                                                                            all_K_fold_models)
        knn_model = self.knn.train(train_imgs, incorrect_labels)
        model_easy, model_hard = self.emotion_classifier.run(knn_model, train_imgs, train_labels)
        self.save_data_models(train_imgs, train_labels, val_imgs, val_labels, knn_model, model_easy, model_hard)

    @staticmethod
    def save_data_models(train_imgs, train_labels, val_imgs, val_labels, knn_model, model_easy, model_hard):
        saveNumpys(train_imgs, train_labels, val_imgs, val_labels)
        saveKnnModel(knn_model, "models", "knn_models.joblib")
        saveModel(model_easy, "models", "model_easy.pt")
        saveModel(model_hard, "models", "model_hard.pt")

class RunEvaluation:
    def __init__(self, class_names):
        self.evaluator = Evaluator(class_names)

    def run_evaluation(self):
        self.evaluator.runEvaluation("./models/model_easy.pt",
                                     "./models/model_hard.pt",
                                     "./models/knn_models.joblib",
                                     "./numpy_data/val_imgs.npy",
                                     "./numpy_data/val_labels.npy")

if __name__ == '__main__':
    dir = 'CK+'
    class_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']
    # comment out these two lines to disable training
    train = RunTraining(dir)
    train.run_training()

    evaluate = RunEvaluation(class_names)
    evaluate.run_evaluation()
