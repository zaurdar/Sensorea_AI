from process_pipeline import *
import streamlit as st
class LSTM(nn.Module):
    def __init__(self,in_channels,out_channels,hidden=[128,64]):
        super(LSTM, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels,hidden[0]),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden[0],hidden[1],batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden[1],out_channels),
                                  nn.Sigmoid())
    def forward(self, x):
        x = self.mlp(x)
        x, (h_n, c_n) = self.lstm(x)
        x = x[:,-1,:]
        x = self.head(x)
        return x
#appel des fonctions de traitement de l'entrée : elle vont faire exactement ce qui est mis dans le notebook
#pas besoin d'utiliser dans une utilisation ponctuelle c'est pour quand on veut faire un réentrainement
Inputs_pipeline = preprocessing_pipeline()
Inputs_pipeline.load("X_pipeline")#bien vérifier les paths
pipeX = Inputs_pipeline.fit() #création de la fonction

Outputs_pipeline = preprocessing_pipeline()
Outputs_pipeline.load("y_pipeline")
# si on veut modifier les param il faut appeler Outputs_pipeline.tree et changer la clef param pour la fonction qu'on veut changer
pipeY = Outputs_pipeline.fit()

#ne juste pas oublier de faire un torch.tensor().float() juste après
#load pipeline de test on mets X [20,14] dedans et il va convertir les dates
Test_pipeline = preprocessing_pipeline()
Test_pipeline.load("test_pipeline")#bien vérifier les paths
testpipe = Test_pipeline.fit() #création de la fonction

#load du model
param = torch.load("lstm_model.pt")
model_state = param["model_state"]
in_channels = param["in_channels"]
out_channels = param["out_channels"]
# ne pas oublier de définir la classe avant d'import
model_test = LSTM(in_channels,out_channels)
model_test.load_state_dict(model_state)
model_test.eval()

def predict(model,x):
    x = x.unsqueeze(0)
    y_pred = model(x)
    return y_pred

#exemple d'utilisation concrète :
#au final x a une shape 18
#x = [date[1]->datatime,date,Cmd[9]->[0,1],water[1]->float,elec[1]->float,gaz[1]->float,temp[1]->float,occ[1}->[0,1]]
def conv_predict(x,model=model_test,pipe=testpipe):
    X =pipe(x)
    X = torch.tensor(X).float()
    y_pred = predict(model,X)
    return y_pred
def train(dataframe,pipeY=pipeY,pipeX=pipeX,lr=0.5e-4,epoch = 1200):
    X = pipeX(dataframe)
    X = torch.tensor(X).float()
    y = pipeY(dataframe)
    y = torch.tensor(y).float()
    model = LSTM(18, 9)
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.MSELoss()
    compiler_LSTM = pytorch_compiler(model, optimizer, criterion,epoch , 32)
    model, train_losses, val_losses = compiler_LSTM.fit(X, y)
    torch.save({
        "model_state": model.state_dict(),
        "in_channels": 18,
        "out_channels": 9,
        "hidden": (128, 64)
    }, "lstm_model.pt")
    return model, train_losses, val_losses
