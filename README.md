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

<pre><code>
  def scaled_dot_product_attention(inputs, d_k, mask=None, causal=False):
    # Project inputs to query, key, and value vectors
    queries = Dense(d_k)(inputs)
    keys = Dense(d_k)(inputs)
    values = Dense(d_k)(inputs)
    # Compute scaled dot-product of queries and keys
    attention_scores = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(tf.cast(d_k, tf.float32))
    # Apply causal masking if needed
    if causal:
        # Create a lower triangular matrix to mask future timesteps
        # Shape: (batch_size, time_steps, time_steps)
        causal_mask = tf.linalg.band_part(tf.ones_like(attention_scores), -1, 0)
        attention_scores += (1.0 - causal_mask) * -1e9  # Mask future timesteps
    # Apply additional mask if provided (e.g., padding mask)
    if mask is not None:
        attention_scores += (mask * -1e9)  # Mask out padding elements
    # Apply softmax to get the attention weights
    attention_weights = Activation('softmax')(attention_scores)
    # Compute the context vector as a weighted sum of the values
    context_vector = tf.matmul(attention_weights, values)
    return context_vector
  # Encoder LSTM
    encoder_inputs = Input(shape=(time_steps, n_features))
    encoder_LSTM = LSTM(LSTM_unit, activation='relu',return_sequences=True)
    encoder_output = encoder_LSTM(encoder_inputs)
    encoder_outputs = Dense(1)(encoder_output)
    #TCN_model
    x=encoder_outputs
    dilation_rates = [1, 2, 4, 8, 16]# Define a list of dilation rates (adjust as needed)
    one_conv_outputs = Conv1D(TCN_filter, kernel_size=TCN_kernel_size, dilation_rate=1, padding='causal', kernel_initializer='he_uniform')(x) # Initial convolution (outside the loop)
    #Dilated convolutional network
    dil_conv_inputs = one_conv_outputs
    for rate in dilation_rates[1:]:  # Start from the second rate (d=1)
        dilated_conv = Conv1D(TCN_filter, kernel_size=TCN_kernel_size, dilation_rate=rate, padding='causal', kernel_initializer='he_uniform')(dil_conv_inputs)
        normalized_conv = BatchNormalization()(dilated_conv)
        activated_conv = Activation('relu')(normalized_conv)
        dropped_conv = Dropout(drop_out)(activated_conv)
        dilated_conv_1 = Conv1D(TCN_filter, kernel_size=TCN_kernel_size, dilation_rate=rate, padding='causal', kernel_initializer='he_uniform')(dropped_conv)
        normalized_conv_1 = BatchNormalization()(dilated_conv_1)
        activated_conv_1 = Activation('relu')(normalized_conv_1)
        dropped_conv_1 = Dropout(drop_out)(activated_conv_1)
        added = Add()([one_conv_outputs, dropped_conv_1])
        out = Activation('relu')(added)
        AM = scaled_dot_product_attention(out, d_k_value, mask=None)
        dil_conv_inputs = AM
    flattened_outputs = Flatten()(dil_conv_inputs)
    dense_outputs = Dense(fc_neurons, activation='relu')(flattened_outputs)  # Fully connected layer
    tcn_outputs = Dense(1)(dense_outputs)  # Final output layer
</code></pre> 

📁 Dataset
Used datasets per season (CSV format):

Photovoltaic spring 2022–2023

Photovoltaic summer 2022–2023

Photovoltaic autumn 2022–2023

Photovoltaic winter 2022–2023

Each contains:
<pre><code>
timestamp, Active_Power, Wind_Speed, Weather_Temperature_Celsius,
Global_Horizontal_Radiation, Wind_Direction, Weather_Daily_Rainfall,
Max_Wind_Speed, Air_Pressure, Hail_Accumulation
</code></pre>  
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

Each run of the SPDA() function:

  📂 Loads and preprocesses the seasonal dataset

  🧱 Builds the attention-enhanced TCN model with residual blocks

  🏋️ Trains the model and performs validation on the test split

  📊 Evaluates performance

  📉 Plots predicted vs actual power generation curves


📈 Evaluation Metrics
The model reports the following metrics:

  ✅ RMSE: Root Mean Square Error – measures overall prediction error

  📏 MAE: Mean Absolute Error – measures average magnitude of error

  📈 R² Score: Coefficient of Determination – indicates goodness of fit (1.0 is ideal)


📊 Experimental Results
| 🌤️ Season | 📉 RMSE | 📏 MAE | 📈 R² Score |
|-----------|--------:|-------:|------------:|
| Spring    | 0.0576   | 0.0245  | 0.9596    |
| Summer    | 0.025    | 0.0098  | 0.9901    |
| Autumn    | 0.0208   | 0.0091  | 0.9959    |
| Winter    | 0.0394   | 0.0229  | 0.9772    |


📦 Installation
<pre><code>
pip install tensorflow keras pandas scikit-learn matplotlib
</code></pre>

👩‍💻 Author
Nguyen Quynh Anh
International University – Vietnam National University
