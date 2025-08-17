import json
import os
import importlib.util
import pandas as pd
import sys
import autogrow.operators.mutation.execute_mutations as execute_mutations
print("+++++++++Finished Imports+++++++++")


def MyPipeline(vars_file_path) -> None:
    '''
    Inputs
    vars_file_path: a string of your file path to your vars json file
    '''
    # Set vars
    with open(vars_file_path, "r") as f:
        vars = json.load(f)
    
    # dummy vars required by the mutation function
    vars["parallelizer"] = FakeParallelizer()
    vars["filter_object_dict"] = {}

    ligands_list = []
    with open(vars['source_compound_file'], "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:  # Make sure there are exactly two columns
                ligands_list.append(parts)
    print("Created ligands list")
    
    num_mutants_to_make = vars['num_mutants_to_make']

    os.makedirs(vars['output_directory'], exist_ok=True)
    print(f'Created output directory: {vars["output_directory"]}')

    PipelineClass = load_class_from_file(vars['FeaturizeToPredictionPipeline_Path'], vars['FeaturizeToPredictionPipeline_ClassName'])

    # Main loop
    for generation in range(1,vars['number_of_generations']+1):
        print(f"{len(ligand_list)} ligands passed to next round")
        
        print(f"===Started generation {generation}===")
        # Perform Mutations
        new_mutants = execute_mutations.make_mutants(
            vars=vars,
            generation_num=1,
            number_of_processors=1,
            num_mutants_to_make=num_mutants_to_make,
            ligands_list=ligands_list,
            new_mutation_smiles_list=[],
            rxn_library_variables=vars['rxn_library_variables'],
        )
        
        print(f"Created {len(new_mutants)} new mutants")

        df = pd.DataFrame(new_mutants, columns=['SMILES', 'Name'])

        Pipeline= PipelineClass(df['SMILES'], vars)
        df_sorted = Pipeline.featurize_and_score()

        # Save csv
        df_sorted.to_csv(vars["output_directory"] + f"/generation{generation}", sep="\t", index=False)
        print(f'Saved mutations and predictions as generation{generation}')

        # Set ligand_list variable to the top X smiles strings
        ligand_list = df_sorted['SMILES'].head(vars['num_of_smi_pass'])
        print(f"-- Completed Generation: {generation} --")

    print("\n****** Completed ******")
    print(f"Check output in {vars['output_directory']}")


def load_class_from_file(class_file_path, class_name):
    '''
    Inputs
    class_file_path: a string of your file path to your class python file
    class_name: a string of your class name
    '''
    module_name = os.path.splitext(os.path.basename(class_file_path))[0]
    # Load the module spec
    spec = importlib.util.spec_from_file_location(module_name, class_file_path)
    if spec is None:
        raise ImportError(f"Cannot find module at {class_file_path}")
    
    # Load the module
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Class {class_name} not found in {class_file_path}")
    
    return cls


# fake parallelizer to run serially
class FakeParallelizer:
    def run(self, job_input, function_to_run):
        results = []
        for args in job_input:
            if isinstance(args, (list, tuple)):
                result = function_to_run(*args)
            else:
                result = function_to_run(args)
            results.append(result)
        return results
    
    def return_node(self):
        return 1  # pretend we have 1 processor



if __name__ == "__main__":
    pipeline_name = sys.argv[1]
    vars_path = sys.argv[2]

    if pipeline_name == "MyPipeline":
        MyPipeline(vars_path)





