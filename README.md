# Medication Recommendation System for Polypharmacy Patients

A graph neural network (GNN) approach to recommend medications for patients with multiple conditions, incorporating adverse event prediction.

## Project Overview
This project addresses the challenge of medication recommendation for polypharmacy patients (those taking multiple medications). Unlike traditional approaches that focus on single diseases, this system uses graph neural networks to model the complex relationships between patients, medications, and diagnoses.

## Features
- Heterogeneous graph construction using MIMIC-III data
- GNN-based medication recommendation
- Adverse event prediction for medication combinations
- Visualization of patient medication networks
- Diagnosis importance analysis

## Data
This project uses the MIMIC-III clinical database, focusing on:
- Prescription records
- Diagnosis codes
- Patient demographics
- Admission information

Note: You'll need access to MIMIC-III to run this code. Follow the official instructions at [PhysioNet](https://physionet.org/content/mimiciii/1.4/) to request access.

## Technical Implementation
- **Graph Construction**: Building a heterogeneous graph with patients, medications, and diagnoses as nodes
- **Graph Neural Network**: Implementing a GNN to learn representations for all entities
- **Link Prediction**: Formulating medication recommendation as a link prediction task
- **Adverse Event Detection**: Using medication co-occurrence patterns to identify potential adverse interactions

## Requirements
See `requirements.txt` for the full list of dependencies.

## Getting Started
1. Clone this repository
2. Install the required packages: `pip install -r requirements.txt`
3. Place your MIMIC-III CSV files in the `data/` directory
4. Run the Jupyter notebook: `jupyter notebook Medication_Recommendation_System_for_Polypharmacy_Patients.ipynb`

## Results
The model achieves a test AUC of approximately 0.8, demonstrating good performance in recommending appropriate medications for polypharmacy patients.

## Future Work
- Incorporate time-series data to model treatment progression
- Include additional patient features (lab results, vitals)
- Integrate external knowledge bases for more accurate adverse event prediction
- Develop an interpretable interface for healthcare providers

# Data Directory

Place your MIMIC-III CSV files in this directory:
- PRESCRIPTIONS.csv
- DIAGNOSES_ICD.csv
- PATIENTS.csv
- ADMISSIONS.csv

Due to data sharing restrictions and file size limitations, these files are not included in the repository. To obtain the MIMIC-III dataset, request access through PhysioNet: https://physionet.org/content/mimiciii/1.4/

## License
This project is licensed under the MIT License - see the **LICENSE** file for details.
