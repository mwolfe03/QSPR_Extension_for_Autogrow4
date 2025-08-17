# QSPR_Extension_for_Autogrow4

This repository provides a lightweight pipeline that runs only the **click chem mutation scripts** from [Autogrow4](https://github.com/kinase/AutoGrow4), allowing users to build an **evolutionary pipeline** with their own **custom QSPR scoring classes** to optimize ligand structures.  

QSPR_Extension_for_Autogrow4 was developed for **Dr. Eric Sapper's Polymers and Coatings Lab** to integrate Autogrow4’s mutation algorithms with QSPR models.  

---

## 🚀 Usage

Run the `MyPipeline` function from `MyPath2`, where the input is the **file path** to your custom variables JSON file.  

- Example input file: [`MyCustomVar.json`](./MyCustomVar.json)  
- Inside this JSON, include a key pointing to the path of your **custom scoring class**.  

Your custom scoring class should inherit from:  

- **Parent Class**: `FeaturizeToPredictionPipelineParent.py`  
- **Example Implementation**: `FeaturizeToPredictionPipeline_RFR_RDKitDescriptors.py`

---

## ⚙️ Dependencies

Make sure you have the following installed:

- [AutoGrow4](https://github.com/kinase/AutoGrow4)  
- [NumPy](https://numpy.org/)  
- [Pandas](https://pandas.pydata.org/)  

