from FeaturizeToPredictionPipelineParent import FeaturizeToPredictionPipeline
import deepchem as dc
import pandas as pd
import numpy as np
import pickle

class FeaturizeToPredictionPipeline_RFR_RDKitDescriptors(FeaturizeToPredictionPipeline):
    def __init__(self, smiles_array, vars:dict):
        '''
        Inputs
        mutant_list : list of lists following this order [[smiles1, name1], [smiles2, name2], ...]
        vars: dictionary containing 'model_path' and other user defined variables

        '''
        self.model_path = vars['model_path']
        self.vars = vars
        self.smiles_array = smiles_array
        

    def featurize(self) -> np.ndarray:
        """Convert SMILES strings into features. To be implemented by subclass.

        Inputs
        smiles_array: list or array of all smiles strings
        vars:

        Outputs:
        np.ndarray
        """
        featurizer = dc.feat.RDKitDescriptors()
        features = featurizer.featurize(self.smiles_array)

        # fill empty array with values that will be removed in the imputer step
        for i in range(len(features)):
            
            if len(features[i]) == 0:
                features[i] = np.array([100000.123]*217)
                
        
        return np.array(features) # np.ndarray


    def impute(self, dataset) -> pd.DataFrame:
        """
        Performs any user defined imputing  
        """
        # I will just treat this as a filter rather than an imputer. RDKit descriptors often produces inf or near inf values. 
        # I do not want these molecules making it into the next generation.
        
        df = pd.DataFrame(dataset)

        if len(df.columns) == 1: # I don't know why this works but not more simple options do not. If you can make this more elegant and still works, please do so.
            new_dataset_list = []
            for i in dataset:
                new_dataset_list.append(i)
            
            df = pd.DataFrame(np.array(new_dataset_list))
                
                
        df['SMILES'] = self.smiles_array
        df= df.dropna()
        # Keep only rows where the specified column is within the threshold
        numeric_df = df.drop('SMILES', axis=1)
        
        mask = numeric_df.applymap(lambda x: -10000 <= x <= 10000)
        df_cleaned = df[mask.all(axis=1)]
        return df_cleaned # pd.DataFrame
    
    def transform(self, dataset) -> np.ndarray:
        """
        Performs any user defined transforming
        """
        return dataset

    def predict(self, dataset) -> np.array:
        """Predict the property using the model. To be implemented by subclass.
        Output:
        np.array
        """
        with open(self.vars['model_path'], 'rb') as f:
            RFR_model = pickle.load(f)
        
        prediction = RFR_model.predict(dataset.drop('SMILES', axis=1))
        return prediction
        

    def order(self, scored_dataset: pd.DataFrame) -> pd.DataFrame:
        """Order the molecules"""
        ordered_dataset = scored_dataset.sort_values(by='Prediction', ascending=False)
        return ordered_dataset



    def featurize_and_score(self) -> pd.DataFrame:
        """Featurize → Impute → Transform → Score"""
        np_dataset = self.featurize()
        pd_dataset = self.impute(np_dataset)
        pd_dataset = self.transform(pd_dataset)
        prediction = self.predict(pd_dataset)

        scored_dataset = pd.DataFrame({
            'SMILES': pd_dataset['SMILES'],
            'Prediction': prediction
        })

        ordered_dataset = self.order(scored_dataset)


        return ordered_dataset # pd.DataFrame