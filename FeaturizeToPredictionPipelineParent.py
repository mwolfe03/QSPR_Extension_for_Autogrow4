import numpy as np
import pandas as pd

class FeaturizeToPredictionPipeline:
    def __init__(self, smiles_array, vars:dict):
        '''
        Inputs
        mutant_list : list of lists following this order [[smiles1, name1], [smiles2, name2], ...]
        vars: dictionary containing 'model_path' and other user defined variables

        '''
        self.model_path = vars['model_path']
        self.smiles_array = smiles_array
        self.vars = vars

    def featurize(self, smiles_array):
        """Convert SMILES strings into features. To be implemented by subclass.

        Inputs
        smiles_array: list or array of all smiles strings
        vars:

        Outputs:
        np.ndarray
        """
        raise NotImplementedError("Subclasses must implement 'featurize' method.")

    def impute(self, dataset) -> np.ndarray:
        """
        Performs any user defined imputing
        """
        return dataset
    
    def transform(self, dataset) -> np.ndarray:
        """
        Performs any user defined transforming
        """
        return vars

    def predict(self, dataset) -> np.array:
        """Predict the property using the model. To be implemented by subclass.
        
        Output:
        np.array
        """
        raise NotImplementedError("Subclasses must implement 'score' method.")

    def order(self, dataset) -> pd.DataFrame:
        """Order the molecules by """
        raise NotImplementedError("Subclasses must implement 'score' method.")

    def featurize_and_score(self) -> pd.DataFrame:
        """Featurize → Impute → Transform → Score"""
        dataset = self.featurize(self.smiles_array)
        dataset = self.impute(dataset.X)
        dataset = self.transform(dataset)
        prediction = self.predict(dataset)
        scored_dataset = pd.DataFrame({
            'SMILES': self.smiles_array,
            'Prediction': prediction
        })
        ordered_dataset = self.order(scored_dataset)
        return ordered_dataset