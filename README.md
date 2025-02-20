---

# IIT Guwahati AIML Second Round Submission

Welcome to the repository for the **IIT Guwahati AIML Second Round** project by [Sandeep Kommineni](https://github.com/SANDEEPxKOMMINENI). This project focuses on predicting the optimal time slots for sending emails to customers to maximize open rates, leveraging a dataset of customer communication history and additional features.

## Project Overview

The goal of this project is to develop a machine learning model that predicts the best time slots (out of 28 possible slots spanning Monday to Sunday, 9:00 AM to 9:00 PM) for sending emails to customers. The model is trained on historical customer interaction data and uses a neural network with focal loss to handle class imbalance and improve prediction accuracy.

### Key Features
- **Dataset**: Utilizes customer communication history (`train_action_history.csv`) and additional customer data (`train_cdna_data.csv`) from a Kaggle dataset.
- **Preprocessing**: Extensive data cleaning, feature engineering, normalization, and encoding.
- **Model**: A deep neural network with batch normalization, dropout, and Swish activation, optimized using focal loss.
- **Output**: Predicts the top time slots for each customer in the test set, exported as `final_submission.csv`.

## Repository Structure

```

All CSV Files are not included due to size issues. Huge dataset
IIT-Guwahati-AIML-Second-Round/                      
│
├── data/                    # Directory for datasets (not included in repo due to size)
│   ├── train_action_history.csv
│   ├── train_cdna_data.csv
│   ├── test_action_history.csv
│   ├── test_cdna_data.csv
│   └── test_customers.csv
│
├── Final_TrainDataset.csv   # Processed training dataset
├── FinalModel.keras         # Trained neural network model
├── label_encoders.pkl       # Label encoders for categorical features
├── numerical_column_means.csv # Means of numerical columns for imputation
├── relevant_columns_for_finalmodel.csv # Columns used in the final model 
├── final_submission.csv     # Final predictions for test customers
│
├── iit-guwahati-aiml-second-round.ipynb # Main Jupyter notebook with all code
├── README.md                # This file
└── requirements.txt         # List of required Python packages
```

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Kaggle API (for dataset download)
- Magic Wormhole (optional, for file transfer)

Install the required packages:
```bash
pip install -r requirements.txt
```

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SANDEEPxKOMMINENI/IIT-Guwahati-AIML-Second-Round.git
   cd IIT-Guwahati-AIML-Second-Round
   ```

2. **Download the Dataset**:
   - Use Kaggle API to download the dataset:
     ```bash
     kaggle datasets download -d sandeepk0mmineni/team-winners-dataset
     unzip team-winners-dataset.zip -d data/
     ```
   - Alternatively, manually download from [Kaggle](https://www.kaggle.com/datasets/sandeepk0mmineni/team-winners-dataset) and place files in the `data/` directory.

3. **Run the Notebook**:
   - Open `iit-guwahati-aiml-second-round.ipynb` in Jupyter Notebook or JupyterLab:
     ```bash
     jupyter notebook iit-guwahati-aiml-second-round.ipynb
     ```
   - Execute the cells sequentially to preprocess data, train the model, and generate predictions.

## Data Preprocessing

- **Action History**: Contains 6 months of interaction data for 200,000 customers (8.7M rows). Features include `customer_code`, `Offer_id`, `send_timestamp`, and `open_timestamp`.
- **CDNA Data**: Additional customer data with 12.85M rows and 303 columns.
- **Cleaning**: Removed columns with >40% missing values, applied variance thresholding, and imputed missing values with means.
- **Feature Engineering**: Added `is_opened`, `send_day_of_week`, `send_hour_of_day`, and time slot features (`slot_1` to `slot_28`).
- **Normalization**: Applied MinMaxScaler to numerical features.
- **Encoding**: Used LabelEncoder for categorical variables.

## Model Training

- **Model Architecture**: A deep neural network with:
  - Batch normalization
  - Dense layers (512, 384, 256 units) with Swish activation
  - Dropout (0.5, 0.4, 0.3)
  - Output layer with softmax for 28 classes
- **Loss Function**: Custom focal loss to address class imbalance.
- **Optimizer**: AdamW with learning rate scheduling.
- **Metrics**: Accuracy and Top-3 Accuracy.
- **Training**: 80-20 train-test split, 200 epochs with early stopping.

## Prediction

- **Test Data**: Processed `test_action_history.csv`, `test_cdna_data.csv`, and `test_customers.csv`.
- **Output**: Predicted time slot orders for each customer in `final_submission.csv`.
- **Format**: 
  ```
  customer_code,predicted_slots_order
  CUST000001,"['slot_15', 'slot_12', 'slot_8', ...]"
  ...
  ```

## Evaluation

- **Test Accuracy**: 0.85
- **Top-3 Accuracy**: 0.92

## Usage

To generate predictions for new data:
1. Ensure test data matches the format of `test_action_history.csv` and `test_cdna_data.csv`.
2. Load `FinalModel.keras`, `label_encoders.pkl`, and `numerical_column_means.csv`.
3. Preprocess the data as per the notebook.
4. Run predictions using the loaded model.

## File Transfer

Some files (e.g., `FinalModel.keras`) were transferred using Magic Wormhole:
```bash
wormhole send FinalModel.keras
wormhole receive <code>  # Replace <code> with the received code
```

## Acknowledgments

- IIT Guwahati AIML Team for organizing the competition.
- Kaggle for hosting the dataset at [https://www.kaggle.com/datasets/sandeepk0mmineni/team-winners-dataset](https://www.kaggle.com/datasets/sandeepk0mmineni/team-winners-dataset).
- TensorFlow and scikit-learn communities for excellent tools.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, reach out to [Sandeep Kommineni](mailto:kvkkbabu@gmail.com) or open an issue on this repository.

---
