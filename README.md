📊 SPDA: Scaled Dot-Product Attention in Temporal Convolutional Residual Blocks for Photovoltaic Forecasting
This repository presents SPDA, a hybrid deep learning model tailored for time series forecasting of photovoltaic (PV) power generation. The key innovation of SPDA lies in the integration of Scaled Dot-Product Attention within Temporal Convolutional Network (TCN) residual blocks, allowing the model to dynamically focus on the most relevant time steps at different receptive fields.

🔬 Key Highlights
Hybrid Architecture: Combines LSTM, TCN, and attention mechanisms.

Attention-Enhanced TCN: Injects Scaled Dot-Product Attention inside each TCN residual block to enhance temporal feature selection.

Seasonal Modeling: Trained and evaluated on seasonal datasets (Spring, Summer, Autumn, Winter) for robust performance analysis.

🧠 Model Architecture Overview
LSTM Encoder – Encodes the input time series.

Conv1D Layer – Captures short-term trends.

Residual TCN Blocks with Attention – Each TCN residual block integrates scaled dot-product attention to focus on key patterns.

Dense Layers – Final prediction stage.

📁 Dataset
Used datasets per season (CSV format):

Photovoltaic spring 2022–2023

Photovoltaic summer 2022–2023

Photovoltaic autumn 2022–2023

Photovoltaic winter 2022–2023

Each contains:

sql
Copy
Edit
timestamp, Active_Power, Wind_Speed, Weather_Temperature_Celsius,
Global_Horizontal_Radiation, Wind_Direction, Weather_Daily_Rainfall,
Max_Wind_Speed, Air_Pressure, Hail_Accumulation
Hail_Accumulation is removed during preprocessing.

⚙️ Key Parameters
<pre><code> time_steps = 3 
  n_features = 8 
  TCN_kernel_size = 4 
  Epoch_value = 70 
  Batch_size_value = 72 
  learning_rate = 0.001 
  fc_neurons = 10 GRU_unit = 64 
  d_k_value = 128 drop_out = 0.2 
  LSTM_unit = 32 
  TCN_filter = 32 </code></pre>

▶️ How to Run
Each seasonal model is trained by calling:
<pre><code>
SPDA(spring1, spring2, spring3)
SPDA(summer1, summer2, summer3)
SPDA(autumn1, autumn2, autumn3)
SPDA(winter1, winter2, winter3)
  </code></pre>
  
Each run:

Loads and processes the data

Builds the attention-enhanced TCN model

Trains and validates the model

Evaluates and plots predicted vs actual values

📈 Evaluation Metrics
RMSE: Root Mean Square Error

MAE: Mean Absolute Error

R² Score: Coefficient of Determination

Also includes line plots of predictions vs actual power output.

📦 Installation
bash
Copy
Edit
pip install tensorflow keras pandas scikit-learn matplotlib


👩‍💻 Author
Nguyen Quynh Anh
International University – Vietnam National University
