# Drug Legitimacy Verification Using Machine Learning and Blockchain

This project is designed to verify the legitimacy of drug batches using a combination of **Machine Learning models**, **Blockchain smart contracts**, and **QR code verification**.  
It was developed as part of an academic MCA final year project.

---

## 📌 Features
- **Image Verification**  
  - Autoencoder pre-filter rejects invalid inputs.  
  - CNN (`cnn_best_model.h5`) classifies drug package images as Legitimate, Fake, or Invalid.  

- **Structured Input Verification**  
  - Random Forest model (uploaded as a Release asset due to large size).  
  - Takes structured drug data (manufacturer, salt, type, etc.) and predicts legitimacy.  

- **Blockchain Integration**  
  - Smart contract (`DrugVerification.sol`) deployed on Ganache/Ethereum test network.  
  - Stores registered batches and verifies legitimacy.  

- **QR Code Support**  
  - Each drug batch has a QR code.  
  - The system verifies QR codes via upload or webcam scanner.  

- **Email Alerts**  
  - Sends alerts if a fake drug is detected.  

---

## 📂 Repository Contents
- `app.py` → Main Streamlit app.  
- `cnn_best_model.h5` → Trained CNN model for drug image classification.  
- `autoencoder_model.h5` → Autoencoder for invalid image detection.  
- `encoder_*.pkl` → Encoders for categorical structured data.  
- `DrugVerification.sol` → Solidity smart contract.  
- `contract_abi.txt`, `contract_address.txt` → Blockchain deployment files.  
- `register_batches.py` → Script to register drug batches on blockchain.  
- `requirements.txt` → Required Python packages.  
- `train_*` files → Model training scripts.  
- `medicine_with_batch_ids_10k.csv` → Sample batch dataset.  
- Random Forest model (`rf_model.pkl`, ~750MB) is uploaded as a **Release asset**.  

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/samhit24/Drug_Legitimacy_verification_Using_ML_and_Blockchain.git
cd Drug_Legitimacy_verification_Using_ML_and_Blockchain


2. Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows

3. Install dependencies
pip install -r requirements.txt

4. Download the Random Forest model

Since the file is too large for the repo, download it from the Releases section.
Place it in the root folder of the project.

5. Run the app
streamlit run app.py

🛠️ Tech Stack

Machine Learning: TensorFlow, scikit-learn

Blockchain: Solidity, Web3.py, Ganache

Web App: Streamlit

Other: QR code generation & scanning, Email alerts

📌 Notes

CNN and Autoencoder are included in the repo.

Random Forest model must be downloaded from the Release section.

Smart contract requires Ganache (or any Ethereum testnet).

🎓 Project Info

Title: Drug Legitimacy Verification Using Machine Learning and Blockchain
Developer: Samhit Nayak
Course: MCA Final Year Project
