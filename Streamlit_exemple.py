import streamlit as st
from model_utils import *
import matplotlib.pyplot as plt
st.title("Exemple d'appel Streamlit")
dummy = torch.load("dummy.pt", weights_only=False)
st.write(dummy["dataframe"])

full_database = dummy["dataframe"]
routine_data = dummy["dataframe"][0:20]

if st.button("prediction"):
    st.write("window : ")
    st.write(routine_data)
    st.write("prediction : ")
    st.write(conv_predict(routine_data))
if st.button("training"):
    with st.spinner("training en cours..."):
        model, train_losses, val_losses = train(full_database)
    st.write("training succesfully processed")
    #pas obligatoire mais je mets pcq c'est joli
    fig, ax = plt.subplots()
    x = [i for i in range(len(train_losses))]
    LSTM_RMSE_val = np.sqrt(val_losses)
    LSTM_Std_val = LSTM_RMSE_val / np.sqrt(2 * 180 * 9)
    LSTM_RMSE_train = np.sqrt(train_losses)
    ax.plot(x, LSTM_RMSE_val, label="LSTM validation")
    ax.plot(x, LSTM_RMSE_train, label="LSTM training")
    ax.scatter(len(val_losses) - 1, LSTM_RMSE_val[-1], label=f"LSTM final RMSE : {LSTM_RMSE_val[-1]}", color="black")
    ax.scatter([], [], label=f"LSTM final standard deviation : {LSTM_Std_val[-1]}")
    ax.set_ylabel("RMSE")
    ax.legend()
    st.pyplot(fig)
    fig2, ax2 = plt.subplots()
    def gaussian(x, mu=0.0, sigma=1.0):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


    for i in range(0, len(LSTM_RMSE_val), 109):
        x = np.linspace(0, 0.35, 200)
        y = gaussian(x, mu=LSTM_RMSE_val[i], sigma=LSTM_Std_val[i])
        ax2.plot(x, y, label=f"epoch : {i}")
    ax2.legend()
    st.pyplot(fig2)


