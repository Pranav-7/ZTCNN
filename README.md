# ZTCNN

Paper Title: ZTML: A Zero-Knowledge and TEE-Enabled Framework for Cryptographically Verifiable Machine Learning Inference

Paper Status: In Submission

- **API Reference: Execute and Verify**
    
    
    | **Step** | **What It Does** | **Method** | **Endpoint** |
    | --- | --- | --- | --- |
    | **0** | Check if the API service is running | GET | `/health` |
    | **1** | Create a sample ML model and export it to ONNX | POST | `/step/1` |
    | **2** | Generate EZKL configuration settings | POST | `/step/2` |
    | **3** | Calibrate the model for ZK execution | POST | `/step/3` |
    | **4** | Compile the zero-knowledge circuit | POST | `/step/4` |
    | **5** | Generate the structured reference string (SRS) | POST | `/step/5` |
    | **6** | Produce the circuit witness | POST | `/step/6` |
    | **7** | Set up proving and verification keys | POST | `/step/7` |
    | **8** | Generate the zkSNARK proof | POST | `/step/8` |
    | **9** | Verify the proof | POST | `/step/9` |
    
    **Base URL:** `https://229371d3627516d0946549a5f04b00c8b6dec768-8000.dstack-pha-prod7.phala.network`