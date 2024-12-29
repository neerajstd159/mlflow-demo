import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import pickle

class testModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        # setup dagshub
        dagshub_token = os.getenv('DAGSHUB_PAT')
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT env variable not set")
        
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        
        dagshub_url = "https://www.dagshub.com"
        repo_owner = "neerajstd159"
        repo_name = "mlflow-demo"

        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        # Load the new model from MLflow model registry

        cls.model_name = "my_model"
        cls.latest_version = cls.get_latest_model_version(cls.model_name)
        cls.model_uri = f"models:/{cls.model_name}/{cls.latest_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # Load the vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl'), 'rb')
        
        # Load test data
        cls.test_data = pd.read_csv('data/processed/test_bow.csv')

    @staticmethod
    def get_latest_model_version(model_name, stages=["Staging"]):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=stages)
        return latest_version[0].version if latest_version else None
    
    def model_loading_test(self):
        self.assertIsNotNone(self.model)

    def model_signature_test(self):
        # create a dummy input to test expected input
        input_text = "I like this movie"
        transformed_text = self.vectorizer.transform([input_text])
        test_df = pd.DataFrame(transformed_text.toarray(), columns=[str(i) for i in range(transformed_text.shape[1])])

        # prediction
        prediction = self.model.predict(test_df)

        # verify the input shape
        self.assertEqual(test_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # verify output shape
        self.assertEqual(len(prediction), test_df.shape[0])
        
    def test_model_performance(self):
        # extract features and label from test data
        X_test = self.test_data.iloc[:, :-1]
        y_test = self.test_data.iloc[:, -1]

        # predict
        y_pred = self.model.predict(X_test)

        # calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # expected threshold
        exp_accuracy = 0.7
        exp_precision = 0.7
        exp_recall = 0.7
        exp_f1 = 0.7

        # Assert that the new model meets the performance thresholds
        self.assertGreater(accuracy, exp_accuracy, f"accuracy must be greater than {exp_accuracy}")
        self.assertGreater(precision, exp_precision, f"precision must be greater than {exp_precision}")
        self.assertGreater(recall, exp_recall, f"recall must be greater than {exp_recall}")
        self.assertGreater(f1, exp_f1, f"f1 score must be greater than {exp_f1}")


if __name__ == "__main__":
    unittest.main()