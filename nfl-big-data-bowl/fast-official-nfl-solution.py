"""NFL Big Data Bowl 2020 solution.

This is the kernel I used for my final Kaggle submission. It is built to run
in the nflrush competition environment on Kaggle, not as a regular local script.
"""

# packages
import numpy as np, pandas as pd, scipy as sp
import warnings, os, datetime, gc, time, random, torch
import matplotlib.pyplot as plt

# sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold

# keras
import keras.callbacks as keras_call
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Model

# keras layers
from keras.layers import Input, Dense, Concatenate
from keras.layers import GaussianDropout, GaussianNoise, BatchNormalization

# torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


# get euclidean distance
def euclidean_distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

# get direction of back
def back_direction(orientation):
    if orientation > 180.0: return 1
    else:                   return 0

def strtofloat(x):
    try:    return float(x)
    except: return -1

def sigmoid(x,a,b):
    return sp.special.expit(a*(b-x))

def get_ypred_score(y_pred,y_true):
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
warnings.filterwarnings("ignore")

try:
    from kaggle.competitions import nflrush
    env = nflrush.make_env()
    iter_test = env.iter_test()
except Exception as e:
    print(str(e))

# Softmax layer output transformation
def keras_transform(y):
    return np.clip(np.cumsum(y, axis=1), 0, 1)

# Sigmoid layer output transformation
def pytorch_transform(preds):
    y_pred = preds.copy()
    adjust_preds = np.zeros((len(y_pred), y_pred.shape[1]))
    for idx, pred in enumerate(y_pred):
        prev = 0
        for i in range(len(pred)):
            if pred[i]<prev:
                pred[i]=prev
            prev=pred[i]
        adjust_preds[idx, :] = (pred-np.min(pred))/(np.max(pred)-np.min(pred))
    adjust_preds[:, -1] = 1
    adjust_preds[:, 0] = 0
    return adjust_preds

def get_model_score(model,x,y):
    y_pred  = model.predict(x)
    y_true  = np.clip(np.cumsum(y, axis=1), 0, 1)
    y_pred  = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])

class CRPSCallback(Callback):
    
    def __init__(self,validation, predict_batch_size=20, include_on_batch=False):
        super(CRPSCallback, self).__init__()
        self.validation = validation
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('CRPS_score_val' in self.params['metrics']):
            self.params['metrics'].append('CRPS_score_val')

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['CRPS_score_val'] = float('-inf')

    def on_epoch_end(self, epoch, logs={}):
        logs['CRPS_score_val'] = float('-inf')
            
        if (self.validation):
            X_valid, y_valid = self.validation[0], self.validation[1]
            val_s =  get_model_score(self.model,X_valid,y_valid)
            val_s = np.round(val_s, 6)
            logs['CRPS_score_val'] = val_s

def get_model(TR_DATA,TR_TARGET,VAL_DATA,VAL_TARGET, bsz = 1024, nfold=1):
    
    # setup column space & input layers
    LEN0, LEN1 = TR_DATA[0].shape[1], TR_DATA[1].shape[1]
    inp0, inp1 = Input(shape = (LEN0,)), Input(shape = (LEN1,))
    
    # define hidden layer one
    x0 = Dense(512, input_dim=LEN0, activation='relu')(inp0)
    x1 = Dense(512, input_dim=LEN1, activation='selu')(inp1)
    
    # Augment the static hidden Layer 1
    x0 = GaussianDropout(0.375)(x0)
    x0 = BatchNormalization()(x0)
    
    # Augment the player intuition hidden layer 1
    x1 = GaussianNoise(0.7)(x1)
    x1 = BatchNormalization()(x1)
    x1 = GaussianDropout(0.25)(x1)
    x1 = BatchNormalization()(x1)
    
    # concatentate the static and intution hidden layers
    x = Concatenate(axis=1)([x0,x1])
    
    # pass the concatenated layers through hidden layer 2
    x = Dense(256, activation='relu')(x)
    
    # apply gaussian drop to hidden layer 2
    x = GaussianDropout(0.5)(x)
    x = BatchNormalization()(x)
    
    # pass data to hidden layer hidden layer 3
    x = Dense(256, activation='sigmoid')(x)
    
    # augment model results
    x = GaussianDropout(0.5)(x)
    x = BatchNormalization()(x)
    
    # pass results to the output layer
    out = Dense(199, activation='softmax')(x)
    model = Model([inp0,inp1],out)
    
    # setup model call backs
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
    es = keras_call.EarlyStopping(monitor='CRPS_score_val',  mode='min',  restore_best_weights=True, verbose=False,  patience=10)
    mc = ModelCheckpoint('best_model.h5',monitor='CRPS_score_val',mode='min', save_best_only=True, verbose=False, save_weights_only=True)
    
    # train model
    model.fit(TR_DATA, TR_TARGET,callbacks=[CRPSCallback(validation = (VAL_DATA,VAL_TARGET)),es,mc], epochs=100, batch_size=bsz,verbose=False)
    model.load_weights("best_model.h5")
    
    # calculate feature scores
    tr_s  = np.round(get_model_score(model,TR_DATA,TR_TARGET),6)
    val_s = np.round(get_model_score(model,VAL_DATA,VAL_TARGET),6)
    
    return model,val_s,tr_s

# pytorch Neural network
class NFL_NN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        
        # first layer
        self.fc1 = nn.Linear(in_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.CELU()
        
        # second layer
        self.dout2 = nn.Dropout(0.5)
        self.lin2    = nn.Linear(256,512)
        self.relu2 = nn.ReLU()
        
        self.dout3 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        
        self.dout4 = nn.Dropout(0.25)
        self.out = nn.Linear(256, 199)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        
        # input & first layer
        a1  = self.fc1(input_)
        bn1 = self.bn1(a1)
        h1  = self.relu1(bn1)
        
        d2 =self.dout2(h1)
        f2 = self.lin2(d2)
        a2 = self.relu2(f2)
        
        d3 = self.dout3(a2)
        a3 = self.fc3(d3)
        h3 = self.relu3(a3)
        
        d4 = self.dout4(h3)
        a5 = self.out(d4)
        y = self.out_act(a5)
        return a5
    
class EarlyStoppingIV:
    def __init__(self, patience=2, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, save_name):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_name)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_name):
        torch.save(model.state_dict(), save_name)
        self.val_loss_min = val_loss
        
        
# pytorch model evaluation
def model_eval(model, dataset, data_loader, batch_size):
    model.eval()
    preds = np.zeros((len(dataset), 199))
    with torch.no_grad():
        for i, eval_x_batch in enumerate(data_loader):
                eval_values = eval_x_batch[0].float()
                pred = model(eval_values)
                preds[i * batch_size:(i + 1) * batch_size] = pred
    return preds

def generate_dataloader(df, y_val, batch_size, tr_ix, val_ix):
    
    # create training data & datasets
    tr_x, tr_y = torch.from_numpy(df.iloc[tr_ix].values), torch.from_numpy(y_val[tr_ix])
    val_x, val_y = torch.from_numpy(df.iloc[val_ix].values), torch.from_numpy(y_val[val_ix] )
    tr_dataset, val_dataset = TensorDataset(tr_x, tr_y), TensorDataset(val_x, val_y)

    # create data loaders
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return tr_dataset, val_dataset, tr_loader, val_loader


def train_pytorch_model(train_dataset, valid_dataset, train_loader, valid_loader,torch_feat_len,epoch,batch_size,n_fold):
    
    # setup model
    early_stopping = EarlyStoppingIV(patience=15, verbose=False)
    model          = NFL_NN(torch_feat_len)
    criterion      = nn.MSELoss()
    optimizer      = torch.optim.Adam(model.parameters(), lr=0.003)
    out_features   = 199
    
    # loop through each epoch
    for idx in range(epoch):
        train_batch_loss_sum = 0
        for param in model.parameters():
            param.requires_grad = True

        model.train()
        for x_batch, y_batch in train_loader:
            y_pred = model(x_batch.float())
            loss = torch.sum((y_pred.float()-y_batch.view((len(y_batch), out_features)).float()).pow(2))/(199*len(y_pred))
            train_batch_loss_sum += loss.item()

            del x_batch
            del y_batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
            gc.collect()

        train_epoch_loss = train_batch_loss_sum / len(train_loader)
        valid_y_pred = model_eval(model, valid_dataset, valid_loader, batch_size)
        valid_crps = np.sum(np.power(valid_y_pred - valid_dataset[:][1].data.cpu().numpy(), 2))/(199*len(valid_dataset))
        model_save_name = 'checkpoint_fold_{}.pt'.format(n_fold+1)
        early_stopping(valid_crps, model, model_save_name)
        
        
        nfl_pred   = pytorch_transform(model_eval(model, train_dataset, train_loader, batch_size))
        train_crps = np.sum(np.power(nfl_pred     - train_dataset[:][1].data.cpu().numpy(), 2))/(199*len(train_dataset))
        
        if early_stopping.early_stop:
            break

    # create model & load model with weights
    model = NFL_NN(torch_feat_len)
    model.load_state_dict(torch.load(model_save_name))

    nfl_pred       = pytorch_transform(model_eval(model, train_dataset, train_loader, batch_size))
    valid_y_pred   = pytorch_transform(model_eval(model, valid_dataset, valid_loader, batch_size))
    
    valid_crps = np.sum(np.power(valid_y_pred - valid_dataset[:][1].data.cpu().numpy(), 2))/(199*len(valid_y_pred))
    train_crps = np.sum(np.power(nfl_pred     - train_dataset[:][1].data.cpu().numpy(), 2))/(199*len(train_dataset))
    
    print('Offical Epoch Loss: {:.5f}, Valid CRPS: {:.5f}, Train CRPS: {:.5f}'.format(train_epoch_loss, valid_crps,train_crps))
    del criterion, optimizer
    gc.collect()
    
    return model

def angleAddDeg(x,y):
    if (x > 0)    & (y >= 0): return 0
    elif (x < 0)  & (y <= 0): return 180
    elif (x < 0) & (y > 0):   return -180
    elif (x > 0) & (y < 0):  return -360
    elif (x == 0) & (y > 0):  return 90
    elif (x == 0) & (y > 0):  return 270
    else: return np.nan

def getAngleDeg(x,y):
    if (y == 0) | (x == 0): deg = 0
    else: deg = abs(np.degrees(np.arctan(y/x)))
    deg = abs(deg + angleAddDeg(x,y))
    return deg

def angleDiff(x1,y1,x2,y2):
    vec = np.array([getAngleDeg(x1,y1),getAngleDeg(x2,y2)])
    MAX, MIN = np.max(vec), np.min(vec)
    if (MAX - MIN) <= 180: return MAX - MIN
    else: return 360 - MAX + MIN
    
def rotate_angle_180(x):
    if x < 180: return x + 180
    else:       return  360 - (x + 180)
    
def absoluteAngle(angle):
    # map over angle
    if angle > 180: return 360 - angle
    else: return angle

# Calculate new yard line
def new_line(rush_team, field_position, yardline):
    if rush_team == field_position: return yardline + 0.0 
    else: return 100.0 - yardline # half the field plus the yards between midfield and the line of scrimmage
    
# update yard line
def update_yardline(df):
    new_yardline = df[df['NflId'] == df['NflIdRusher']]
    new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: new_line(x[0],x[1],x[2]), axis=1)
    return new_yardline[['GameId','PlayId','YardLine']]

# Rotate 90 deg based on year
def rotate_angle_90(angle,year):
    if year != 2017: return angle
    if angle > 270:  return 90 - 360 + angle
    else:            return angle + 90

# flip x coordinates
def new_X(x_coordinate, play_direction):
    if play_direction == 'left': return 110.0 - x_coordinate
    else:                        return x_coordinate - 10.0

# flip y coordinates
def new_YNEW(y_coordinate, play_direction):
    if play_direction == 'left': return 53.3 - y_coordinate
    else:                        return y_coordinate

# Non-actual Angle Transpose
def new_orientation(angle, play_direction):
    if play_direction != 'left': return angle
    new_angle = 360.0 - angle
    if new_angle == 360.0: return 0.0
    return new_angle

# Actual Angle Transpose
def new_orientation_II(angle, play_direction):
    if (angle - 90) <= 0: angleII = 90 - angle
    else: angleII = 360 - angle + 90
    if play_direction == 'left': return (180+angleII) % 360
    else: return angleII
    
# update orientation
def update_orientation(df, yardline):
    df['YNEW']          = df[['Y','PlayDirection']].apply(lambda x: new_YNEW(x[0],x[1]), axis=1)
    df['X']             = df[['X','PlayDirection']].apply(lambda x: new_X(x[0],x[1]), axis=1)
    df['OrientationII'] = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation_II(x[0],x[1]), axis=1)
    df['DirII']         = df[['Dir','PlayDirection']].apply(lambda x: new_orientation_II(x[0],x[1]), axis=1)
    df['Orientation']   = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
    df['Dir']           = df[['Dir','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
    df                  = df.drop('YardLine', axis=1)
    return pd.merge(df, yardline, on=['GameId','PlayId'], how='inner')

def getTimeDelta():
    return .625

# update X variable
def physicsX(x,v,a,angle,dt=0):
    if dt == 0: dt = getTimeDelta()
    projx = np.cos(np.radians(angle))      # get projection on x axis
    return x + v*projx*dt + .5*a*projx*(dt**2) # calculate x' given the span of time dt

# update Y variale
def physicsY(y,v,a,angle,dt=0):
    if dt == 0: dt = getTimeDelta()
    projy = np.sin(np.radians(angle))
    return  y + v*projy*dt + .5*a*projy*(dt**2)

# update velocity
def physicsVx(v,a,angle,dt=0):
    if dt == 0: dt = getTimeDelta()
    projx = np.cos(np.radians(angle))   
    return v*projx + a*projx*dt

# update velocity
def physicsVy(v,a,angle,dt=0):
    if dt == 0: dt = getTimeDelta()
    projy = np.sin(np.radians(angle)) # get projection on y axis
    return v*projy + a*projy*dt

def create_count(df,field):
    df[field + '_Count']  = 0
    df.loc[df['Position'] == field,field + '_Count'] = 1
    df[field + '_Count']  = df.groupby('PlayId')[field + '_Count'].transform('sum')

def personnel_features_II(df):
    for key in ['QB','CB','SS','NT','G','S','WR']:
        create_count(df,key)

def defense_features(df):
    calc = ['X','Y','RusherX','RusherY']
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
    rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']
    defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
    defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
    defense['def_dist_to_back'] = defense[calc].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
    defense = defense.groupby(['GameId','PlayId']).agg({'def_dist_to_back':['min','max','mean','std']}).reset_index()
    defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']
    return defense

def defense_pro_features(df):

    # Create Definsive Dataset
    # ---------------------------------
    
    # get rusher
    meta_data   = ['PlayDirection','Season','YardLine']
    rush_fields = ['GameId','PlayId','RusherTeam','RusherX','RusherY','RushS','RushA','RushDir','RushOrient']
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','YNEW','S','A','DirII','OrientationII']]
    rusher.columns = rush_fields
    
    # create table
    DD = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
    DD = DD[DD['Team'] != DD['RusherTeam']][rush_fields + ['X','YNEW','OrientationII','DirII','S','A'] + meta_data]
    
    # Calculate new future points
    DD['RusherFXV'] = physicsX(DD.RusherX,DD.RushS,DD.RushA,DD.RushDir)
    DD['RusherFYV'] = physicsY(DD.RusherY,DD.RushS,DD.RushA,DD.RushDir)
    DD['RusherOXV'] = physicsX(DD.RusherX,DD.RushS,DD.RushA,DD.RushOrient)
    DD['RusherOYV'] = physicsY(DD.RusherY,DD.RushS,DD.RushA,DD.RushOrient)
    
    # measure rushers distance from yard line & map Defenders Acceleration
    DD['RushDistYardLine'] = DD['RusherFXV'] - DD['YardLine']
    
    # calculate new defensive lineman points
    DD['FXV'] = physicsX(DD.X,DD.S,DD.A,DD.DirII)
    DD['FYV'] = physicsY(DD.YNEW,DD.S,DD.A,DD.DirII)

    DD['RusherFXV'] = physicsX(DD.RusherX,DD.RushS,DD.RushA,DD.RushDir)
    DD['RusherFYV'] = physicsY(DD.RusherY,DD.RushS,DD.RushA,DD.RushDir)
    
    
    # calculate rusher velocity
    DD['RusherVelocityX'] = physicsVx(DD.RushS,DD.RushA,DD.RushDir)
    DD['RusherVelocityY'] = physicsVy(DD.RushS,DD.RushA,DD.RushDir)
    
    # calculate the defensive man's velocity
    DD['VelocityX'] = physicsVx(DD.S,DD.A,DD.DirII)
    DD['VelocityY'] = physicsVy(DD.S,DD.A,DD.DirII)
    
    # calculate change in velocities
    DD['VelocityDeltaX'] = DD['VelocityX'] - DD['RusherVelocityX']
    DD['VelocityDeltaY'] = DD['VelocityY'] - DD['RusherVelocityY']
    DD['VelocityDelta']  = np.sqrt(DD['VelocityDeltaX']**2 + DD['VelocityDeltaY']**2)
    
    # Concept Features
    # ---------------------------

    # get new distance equations
    DD['StartDistance']   = euclidean_distance(DD['X'],DD['YNEW'],DD['RusherX'],DD['RusherY'])
    DD['RunnerHeaded']    = euclidean_distance(DD['FXV'],DD['FYV'],DD['RusherFXV'],DD['RusherFYV'])
    DD['RunnerThinking']  = euclidean_distance(DD['FXV'],DD['FYV'],DD['RusherOXV'],DD['RusherOYV'])
    
    # Calculate Time Till Impact
    # ---------------------------
    DD['TimeTillImpact']   = DD['VelocityDelta'] / (DD['RunnerHeaded'] + .01)
    DD['DistanceEstimate'] = DD['TimeTillImpact']*DD['RusherVelocityX'] + DD['YardLine'] + DD['RusherFXV'] - DD['RusherX']
    
    # Defender Assessing
    # ---------------------------

    # Defensive Frame, looking at Position
    DD['RushStartNormX'] = (DD['RusherX'] - DD['X'])/DD['StartDistance']
    DD['RushStartNormY'] = (DD['RusherY'] - DD['YNEW'])/DD['StartDistance']

    # Defensive Frame, Defender Assessing
    DD['DenfenderLookingX'] = np.cos(np.radians(DD['OrientationII']))
    DD['DefenderLookingY']  = np.sin(np.radians(DD['OrientationII']))

    # Rusher Assessing
    # ---------------------------

    # Rusher Frame, Rusher Position
    DD['DefStartNormX'] = (DD['X']    - DD['RusherX'])/DD['StartDistance']
    DD['DefStartNormY'] = (DD['YNEW'] - DD['RusherY'])/DD['StartDistance']

    # Rusher Frame, Rusher Assessing
    DD['RusherLookingX'] = np.cos(np.radians(DD['RushOrient']))
    DD['RusherLookingY'] = np.sin(np.radians(DD['RushOrient']))
    
    # Defender Engagement
    # ---------------------------

    # Calculate Normalized V Units
    DD['RushGoingNormX'] = (DD['RusherFXV'] - DD['X'])/DD['StartDistance']
    DD['RushGoingNormY'] = (DD['RusherFYV'] - DD['YNEW'])/DD['StartDistance']

    # Calculate Normalized Vector
    DD['DenfenderMovingX'] = np.cos(np.radians(DD['DirII']))
    DD['DefenderMovingY']  = np.sin(np.radians(DD['DirII']))


    # Engagement Orientation
    # --------------------------

    # Get Orientation Angle
    DD['DefenderEngagement'] = DD[['RushGoingNormX','RushGoingNormY','DenfenderMovingX','DefenderMovingY']].apply(lambda x: angleDiff(x[0],x[1],x[2],x[3]),axis=1)
    DD['DefenderAssessing']  = DD[['RushStartNormX','RushStartNormY','DenfenderLookingX','DefenderLookingY']].apply(lambda x: angleDiff(x[0],x[1],x[2],x[3]),axis=1)
    DD['RusherAssessing']    = DD[['DefStartNormX','DefStartNormY','RusherLookingX','RusherLookingY']].apply(lambda x: angleDiff(x[0],x[1],x[2],x[3]),axis=1)
    DD['CloseDistance']      = DD['RunnerHeaded'] - DD['StartDistance']

    # get non future location info
    # -----------------------------
    DD['XMAP'] = DD['X']      - DD['RusherX']
    DD['YMAP'] = DD['YNEW']   - DD['RusherY']
    DD['SMAP'] = abs(DD['S']) - abs(DD['RushS'])
    
    
    # setup dataframe
    full = pd.DataFrame()

    # map defensive players to table
    for name, group in DD.groupby(['PlayId','GameId']):
        
        # create variables
        info = {'RushDistYardLine': group['RushDistYardLine'].values[0],
                'PlayId':           group['PlayId'].values[0],
                'GameId':           group['GameId'].values[0]}
        
        x = 0
        for index, row in group.sort_values('RunnerHeaded').iterrows():
            x = x + 1
            for key in ['SMAP','CloseDistance','DefenderAssessing','RusherAssessing','RunnerHeaded','RunnerThinking','DefenderEngagement','DistanceEstimate','TimeTillImpact']:
                if key+str(x) not in info.keys():
                    info[key+str(x)] = [DD.ix[index,key]]

        # concat full dataframe
        full = pd.concat([full,pd.DataFrame(info)],sort=False).reset_index(drop=True)
    return full

def get_offense_prop(df, prefix = 'premier',alpha=3,addXY = True,dt_array = [.2,.3,.4,.5,.6,.7,.8,.9,1.0]):
    
    # get rusher
    meta_data   = ['PlayDirection','Season','YardLine']
    rush_fields = ['GameId','PlayId','RusherTeam','RusherX','RusherY','RushS','RushA','RushDir','RushOrient']
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','YNEW','S','A','DirII','OrientationII']]
    rusher.columns = rush_fields

    # create table
    GK = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
    
    # player score factors
    GK['PlayerFactor']                                         =  1
    GK.loc[GK['Team']     != GK['RusherTeam'],'PlayerFactor']  = -1
    GK.loc[GK['NflId']    == GK['NflIdRusher'],'PlayerFactor'] =  0
    GK.loc[GK['Position'] == 'QB',             'PlayerFactor'] =  0

    ids   = ['GameId','PlayId']
    new   = []
    rushx = []
    rushy = []
    
    for dt in dt_array:

        # get the field to be added
        added_field = prefix + 'PlayerInfluence' + str(int(dt*10000))
        added_rushx  = prefix + 'RusherX' + str(int(dt*10000))
        added_rushy  = prefix + 'RusherY' + str(int(dt*10000))
        
        # Calculate new future points
        GK[added_rushx] = physicsX(GK.RusherX,GK.RushS,GK.RushA,GK.RushDir,dt=dt)
        GK[added_rushy] = physicsY(GK.RusherY,GK.RushS,GK.RushA,GK.RushDir,dt=dt)

        # calculate new defensive lineman points
        GK['PlayerX'] = physicsX(GK.X,GK.S,GK.A,GK.DirII,dt=dt)
        GK['PlayerY'] = physicsY(GK.YNEW,GK.S,GK.A,GK.DirII,dt=dt)

        # player distance
        GK['PlayerDistance'] = euclidean_distance(GK[added_rushx],GK[added_rushy],GK.PlayerX,GK.PlayerY)
        GK[added_field]      = sigmoid(GK.PlayerDistance,1,alpha)*GK['PlayerFactor']
        
        # append new fields
        rushx.append(added_rushx)
        rushy.append(added_rushy)
        new.append(added_field)

    # take aggregate statistics
    data1 = GK[ids + new].groupby(['GameId','PlayId']).min().reset_index(drop=False)
    data2 = GK[ids + new].groupby(['GameId','PlayId']).max().reset_index(drop=False)
    data3 = GK[ids + new].groupby(['GameId','PlayId']).mean().reset_index(drop=False)
    data4 = GK[ids + new].groupby(['GameId','PlayId']).std().reset_index(drop=False)
    data5 = GK[ids + rushx].groupby(['GameId','PlayId']).mean().reset_index(drop=False)
    data6 = GK[ids + rushy].groupby(['GameId','PlayId']).mean().reset_index(drop=False)
    
    # change names
    data1.columns = ids + [s + 'Min'   for s in new]
    data2.columns = ids + [s + 'Max'   for s in new]
    data3.columns = ids + [s + 'Mu'    for s in new]
    data4.columns = ids + [s + 'Sigma' for s in new]
    data5.columns = ids + [s + 'RushX' for s in rushx]
    data6.columns = ids + [s + 'RushY' for s in rushy]
    
    # merge data
    data = pd.merge(data2,data1,how='left',on=['GameId','PlayId'])
    data = pd.merge(data,data3,how='left', on=['GameId','PlayId'])
    data = pd.merge(data,data4,how='left', on=['GameId','PlayId'])
    
    if addXY:
        data = pd.merge(data,data5,how='left', on=['GameId','PlayId'])
        data = pd.merge(data,data6,how='left', on=['GameId','PlayId'])
        
    return data

def static_features(df):

    # get seconds in a year
    seconds_in_year = 60*60*24*365.25
    
    # Setup Constants
    f1 = ['GameId','PlayId','CB_Count', 'SS_Count', 'NT_Count', 'G_Count', 'S_Count', 'WR_Count']
    f2 = ['PlayerAge','WindSpeed_dense','PlayerHeight_dense','diffScoreBeforePlay','Dir_sin','Dir_cos']
    f3 = ['X','Y','S','A','Dis','Orientation','Dir','YardLine','Quarter','Down','Distance','DefendersInTheBox']
    f4 = ['TimeDelta','OrientationII','is_left','is_home','old_data']
    f5 = ['YNEW','DistToEndZone','OffensePersonnel','DefensePersonnel']
    f6 = []
    
    # Time Calculations & Features
    df['TimeHandoff']     = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df['TimeSnap']        = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df['TimeDelta']       = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

    # calculate key data
    df["is_left"]       = df["PlayDirection"] == "left"
    df["is_home"]       = df["Team"] == "home"
    df["old_data"]      = df["Season"] == 2017
    df['DistToEndZone'] = 100 - df['YardLine']
    
    ## Age Calculations & Features
    df['PlayerHeight_dense'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
    df['PlayerAge']          = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)

    ## WindSpeed Feature
    df['WindSpeed_ob'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
    df['WindSpeed_dense'] = df['WindSpeed_ob'].apply(strtofloat)
    
    

    # Dir Features
    df["Dir_sin"] = df["Dir"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
    df["Dir_cos"] = df["Dir"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
    
    # Create formation
    for Form in ['SHOTGUN','I_FORM','SINGLEBACK','PISTOL','JUMBO']:
        df[Form + '_Formation']                                 = 0
        df.loc[df.OffenseFormation == Form,Form + '_Formation'] = 1
        f6.append(Form + '_Formation')

    ## diff Score
    df["diffScoreBeforePlay"] = df["HomeScoreBeforePlay"] - df["VisitorScoreBeforePlay"]
    static_features = df[df['NflId'] == df['NflIdRusher']][f1+f2+f3+f4+f5+f6].drop_duplicates()
    static_features.fillna(-999,inplace=True)
    return static_features

def getDefensiveState(df):
    
    # Create Defensive Table
    # ------------------------
    
    # Field To Create Rush Table
    meta_data   = ['GameId','PlayId','Team','Xpoint','Ypoint','S','A','DirII','OrientationII']
    rush_fields = ['GameId','PlayId','RusherTeam','XpointRush','YpointRush','RushS','RushA','RushDir','RushOrient']
    add_fields  = ['X','YNEW','NflIdRusher','NflId']

    # X Point
    df['Xpoint'] = physicsX(df.X,df.S,df.A,df.DirII,.6)
    df['Ypoint'] = physicsY(df.YNEW,df.S,df.A,df.DirII,.6)

    # Fill Bad Points
    df['Xpoint'] = np.where(df['Xpoint'].isna(), df['X'],    df['Xpoint'])
    df['Ypoint'] = np.where(df['Ypoint'].isna(), df['YNEW'], df['Ypoint'])

    # get rusher
    rusher = df[df['NflId'] == df['NflIdRusher']][meta_data]
    rusher.columns = rush_fields
    rusher['Score'] = 1

    # create table
    data = pd.merge(df[meta_data+add_fields],rusher,on=['GameId','PlayId'],how='inner')
    data['Score'] = data['Score'].fillna(-1)
    data.loc[data['Team'] != data['RusherTeam'],'Score'] = 1

    # create distance stats
    data['DistanceToRusher'] = np.sqrt((data['Xpoint'] - data['XpointRush'])**2 + (data['Ypoint'] - data['YpointRush'])**2)
    data["DefenseIndex"]     = data.groupby(['GameId','PlayId'])["DistanceToRusher"].rank("dense", ascending=True).astype(np.int)


    # Fill Bad Points (this may have created bad data*)
    data['X']    = np.where(data['NflId'] == data['NflIdRusher'],data['Xpoint'], data['X'])
    data['YNEW'] = np.where(data['NflId'] == data['NflIdRusher'],data['Ypoint'], data['YNEW'])

    # create data table
    data = data[['GameId','PlayId','X','YNEW','DistanceToRusher','DefenseIndex','Score','S']]


    # Compile Feature Set
    # -------------------------
    
    # create full
    full      = rusher[['GameId','PlayId']]
    meta_data = ['GameId','PlayId','X','YNEW']
    def_data  = ['GameId','PlayId','DefenderX','DefenderY']


    for key in np.linspace(1,2,2):
        Defender = data[data['DefenseIndex'] == key][meta_data]
        Defender.columns = def_data

        DST = data.merge(Defender,on    = ['GameId','PlayId'],how='left')
        DST['DistanceToRusher']         = np.sqrt((DST['X'] - DST['DefenderX'])**2 + (DST['YNEW'] - DST['DefenderY'])**2)
        DST['DefensiveInfluence'+str(int(key))] = sigmoid(DST['DistanceToRusher'],1,3)*DST['Score'] 
        DST  = DST.groupby(['GameId','PlayId'])[['DefensiveInfluence'+str(int(key))]].mean().reset_index()
        full = pd.merge(full,DST,on = ['GameId','PlayId'],how = 'left')
        
    # Model Input II Features
    feats = ['DefensiveInfluence1', 'DefensiveInfluence2']

    full['defenderMax'] = full[feats].max(axis=1)
    full['defenderMin'] = full[feats].min(axis=1)
    full['defenderSigma'] = full[feats].std(axis=1)    
    
    return full

def create_features(df, deploy=False):
    
    # get back features
    def back_features(df):
        carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','NflIdRusher','X','Y','Orientation','Dir','YardLine']]
        carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
        carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: back_direction(x))
        carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: back_direction(x))
        carriers = carriers.rename(columns={'X':'back_X','Y':'back_Y'})
        return carriers[['GameId','PlayId','NflIdRusher','back_X','back_Y','back_from_scrimmage','back_oriented_down_field','back_moving_down_field']].copy()

    # get features relative to back
    def features_relative_to_back(df, carriers):
        
        # Get Info
        grpflds = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field']
        calc    = ['X','Y','back_X','back_Y']
        player_distance = df[['GameId','PlayId','NflId','X','Y']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
        player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
        player_distance['dist_to_back'] = player_distance[calc].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
        player_distance = player_distance.groupby(grpflds).agg({'dist_to_back':['min','max','mean','std']}).reset_index()
        player_distance.columns = grpflds+ ['min_dist','max_dist','mean_dist','std_dist']
        return player_distance

    # combine data
    def combine_features(relative_to_back, defense, static, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')
        if deploy: return df
        return pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')
    
    personnel_features_II(df)
    yardline      = update_yardline(df)
    df            = update_orientation(df, yardline)
    newset        = getDefensiveState(df)
    
    # Distance Features
    meta_feats    = get_offense_prop(df)
    meta_feats2   = get_offense_prop(df,prefix='alpha1',alpha=1, addXY=False,dt_array = [.8])
    meta_feats4   = get_offense_prop(df,prefix='alpha7',alpha=7, addXY=False,dt_array = [.2])
    
    back_feats    = back_features(df)
    def_feats_pro = defense_pro_features(df)
    rel_back      = features_relative_to_back(df, back_feats)
    def_feats     = defense_features(df)
    static_feats  = static_features(df)
    basetable     = combine_features(rel_back, def_feats, static_feats, deploy=deploy)
    
    # combine other data
    basetable     = pd.merge(basetable,def_feats_pro,how='left',on=['GameId','PlayId'])
    basetable     = pd.merge(basetable,meta_feats,how='left',on=['GameId','PlayId'])
    basetable     = pd.merge(basetable,meta_feats2,how='left',on=['GameId','PlayId'])
    basetable     = pd.merge(basetable,meta_feats4,how='left',on=['GameId','PlayId'])
    
    basetable     = pd.merge(basetable,newset,how='left',on=['GameId','PlayId'])
    
    for key in [ 'premierRusherX10000RushX','premierRusherX2000RushX','premierRusherX4000RushX','premierRusherX6000RushX','premierRusherX7000RushX','premierRusherX8000RushX']:
        basetable[key] = basetable[key] - basetable['YardLine']
    
    return basetable

    
def process_two(t_):
    
    # calculate radian angle
    radian_angle = (90 - t_['Dir']) * np.pi / 180.0
    
    t_['fe1'] = pd.Series(np.sqrt(np.absolute(np.square(t_.X.values) - np.square(t_.Y.values))))
    t_['fe5'] = t_['S'].values*np.cos(radian_angle) + .5 * t_['A'].values *np.cos(radian_angle)
    t_['fe7'] = np.arccos(np.clip(t_['X'].values / t_['Y'].values, -1, 1))  # N
    t_['fe8'] = np.square(t_['S'].values) + 2 * t_['A'].values * t_['Dis'].values
    t_['fe10'] = np.abs(t_['S'] * np.cos(radian_angle))
    t_['fe11'] = np.abs(t_['S'] * np.sin(radian_angle))
    
    t_['OrientationABS'] = t_['Orientation'].apply(lambda x: absoluteAngle(x))
    t_['Orientation']    = t_['Orientation'].apply(lambda x: rotate_angle_180(x))
    t_['DirABS']         = t_['Dir'].apply(lambda x: absoluteAngle(x))
    t_['Dir']            = t_['Dir'].apply(lambda x: rotate_angle_180(x))
    t_['YREF']           = (t_['Y'] - 26.65).abs()
    return t_

# Import Datasets
train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
outcomes = train[['GameId','PlayId','Yards']].drop_duplicates()

# transform speed and acceleration
train.loc[train.Season==2017,'S'] = train.loc[train.Season==2017,'S']*1.1320096503100632
train.loc[train.Season==2017,'A'] = train.loc[train.Season==2017,'A']*1.1210484653841495

# transform orientation
train['Orientation'] = train.apply(lambda x: rotate_angle_90(x.Orientation,x.Season),axis=1)

# Create Training & Target Data
train_basetable = create_features(train, False)
tr_sf = train_basetable.copy()

cat = ['back_moving_down_field','SHOTGUN_Formation', 'I_FORM_Formation', 
       'SINGLEBACK_Formation', 'PISTOL_Formation', 'JUMBO_Formation']
num = ['DefensiveInfluence1', 'DefensiveInfluence2','defenderMax', 'DefensiveInfluence3', 'DefensiveInfluence4', 
'DefensiveInfluence5','DefensiveInfluence6', 'DefensiveInfluence7', 'DefensiveInfluence8', 'DefensiveInfluence9', 
'DefensiveInfluence10', 'DefensiveInfluence11','premierPlayerInfluence2000Max', 'premierPlayerInfluence3000Max', 
'premierPlayerInfluence4000Max', 'premierPlayerInfluence5000Max',
'premierPlayerInfluence6000Max', 'premierPlayerInfluence7000Max', 
'premierPlayerInfluence8000Max', 'premierPlayerInfluence9000Max', 
'premierPlayerInfluence10000Max', 'premierPlayerInfluence2000Min', 
'premierPlayerInfluence3000Min', 'premierPlayerInfluence4000Min',
'premierPlayerInfluence5000Min', 'premierPlayerInfluence6000Min', 'premierPlayerInfluence7000Min',
'premierPlayerInfluence8000Min', 'premierPlayerInfluence9000Min', 'premierPlayerInfluence10000Min', 
'premierPlayerInfluence2000Mu', 'premierPlayerInfluence3000Mu', 'premierPlayerInfluence4000Mu',
'premierPlayerInfluence5000Mu', 'premierPlayerInfluence6000Mu', 'premierPlayerInfluence7000Mu', 
'premierPlayerInfluence8000Mu', 'premierPlayerInfluence9000Mu', 'premierPlayerInfluence10000Mu', 
'premierPlayerInfluence2000Sigma', 'premierPlayerInfluence3000Sigma', 'premierPlayerInfluence4000Sigma',
'premierPlayerInfluence5000Sigma', 'premierPlayerInfluence6000Sigma', 'premierPlayerInfluence7000Sigma',
'premierPlayerInfluence8000Sigma', 'premierPlayerInfluence9000Sigma', 'premierPlayerInfluence10000Sigma',
'back_from_scrimmage', 'min_dist', 'max_dist', 'mean_dist', 'std_dist', 'def_min_dist', 'def_max_dist', 
'def_mean_dist', 'def_std_dist','X', 'YREF', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'YardLine', 'Distance',
'fe1', 'fe5','fe8', 'fe10', 'fe11','Orientation_sin', 'Orientation_cos', 'PlayerAge','PlayerHeight_dense',
'DefendersInTheBox','TimeDelta','old_data','DirABS','OrientationABS','Dir_sin', 'Dir_cos','SMAP1', 'RushDistYardLine',
'CloseDistance1','DefenderAssessing1',  'RusherAssessing1', 'RunnerHeaded1', 'RunnerThinking1', 'DefenderEngagement1',
'DistanceEstimate1','TimeTillImpact1', 'SMAP2', 'CloseDistance2', 'DefenderAssessing2',
'RusherAssessing2', 'RunnerHeaded2', 'RunnerThinking2', 'DefenderEngagement2','DistanceEstimate2',
'SMAP3', 'CloseDistance3', 'DefenderAssessing3', 'RusherAssessing3', 'RunnerHeaded3', 'RunnerThinking3',
'DefenderEngagement3','DistanceEstimate3', 'SMAP4', 'CloseDistance4', 'DefenderAssessing4', 'RusherAssessing4', 
'RunnerHeaded4', 'RunnerThinking4', 'DefenderEngagement4','DistanceEstimate4', 'SMAP5', 'CloseDistance5', 
'DefenderAssessing5', 'RusherAssessing5', 'RunnerHeaded5', 'RunnerThinking5', 'DefenderEngagement5','DistanceEstimate5', 
'SMAP6', 'CloseDistance6', 'DefenderAssessing6', 'RusherAssessing6', 'RunnerHeaded6', 'RunnerThinking6', 'DefenderEngagement6',
'SMAP7', 'CloseDistance7', 'DefenderAssessing7', 'RusherAssessing7', 'RunnerHeaded7', 'RunnerThinking7', 'DefenderEngagement7',
'SMAP8', 'CloseDistance8', 'DefenderAssessing8', 'RusherAssessing8', 'RunnerHeaded8', 'RunnerThinking8', 'DefenderEngagement8',
'SMAP9', 'CloseDistance9', 'DefenderAssessing9', 'RusherAssessing9', 'RunnerHeaded9', 'RunnerThinking9', 'DefenderEngagement9', 
'SMAP10', 'CloseDistance10', 'DefenderAssessing10', 'RusherAssessing10', 'RunnerHeaded10', 
'SMAP11', 'CloseDistance11', 'DefenderAssessing11', 'RusherAssessing11', 'RunnerHeaded11','premierRusherX2000RushX', 
'premierRusherX3000RushX', 'premierRusherX4000RushX', 'premierRusherX5000RushX', 'premierRusherX6000RushX',
'premierRusherX7000RushX', 'premierRusherX8000RushX','premierRusherX9000RushX','premierRusherX10000RushX', 
'alpha7PlayerInfluence2000Max', 'alpha7PlayerInfluence3000Max',
'alpha7PlayerInfluence4000Max', 'alpha7PlayerInfluence5000Max', 'alpha7PlayerInfluence6000Max', 
'alpha7PlayerInfluence7000Max', 'alpha7PlayerInfluence8000Max', 'alpha7PlayerInfluence9000Max',
'alpha7PlayerInfluence10000Max', 'alpha7PlayerInfluence2000Min', 'alpha7PlayerInfluence3000Min',
'alpha7PlayerInfluence4000Min', 'alpha7PlayerInfluence5000Min', 'alpha7PlayerInfluence6000Min',
'alpha7PlayerInfluence7000Min', 'alpha7PlayerInfluence8000Min', 'alpha7PlayerInfluence9000Min', 
'alpha7PlayerInfluence10000Min', 'alpha7PlayerInfluence2000Mu', 'alpha7PlayerInfluence3000Mu', 
'alpha7PlayerInfluence4000Mu', 'alpha7PlayerInfluence5000Mu', 'alpha7PlayerInfluence6000Mu', 
'alpha7PlayerInfluence7000Mu', 'alpha7PlayerInfluence8000Mu', 'alpha7PlayerInfluence9000Mu', 
'alpha7PlayerInfluence10000Mu', 'alpha7PlayerInfluence2000Sigma', 'alpha7PlayerInfluence3000Sigma',
'alpha7PlayerInfluence4000Sigma', 'alpha7PlayerInfluence5000Sigma', 'alpha7PlayerInfluence6000Sigma',
'alpha7PlayerInfluence7000Sigma', 'alpha7PlayerInfluence8000Sigma', 'alpha7PlayerInfluence9000Sigma',
'alpha7PlayerInfluence10000Sigma', 
'alpha5PlayerInfluence2000Max', 'alpha5PlayerInfluence3000Max', 'alpha5PlayerInfluence4000Max',
'alpha5PlayerInfluence5000Max', 'alpha5PlayerInfluence6000Max', 'alpha5PlayerInfluence7000Max',
'alpha5PlayerInfluence8000Max', 'alpha5PlayerInfluence9000Max', 'alpha5PlayerInfluence10000Max',
'alpha5PlayerInfluence2000Min', 'alpha5PlayerInfluence3000Min', 'alpha5PlayerInfluence4000Min', 
'alpha5PlayerInfluence5000Min', 'alpha5PlayerInfluence6000Min', 'alpha5PlayerInfluence7000Min', 
'alpha5PlayerInfluence8000Min', 'alpha5PlayerInfluence9000Min', 'alpha5PlayerInfluence10000Min', 
'alpha5PlayerInfluence2000Mu', 'alpha5PlayerInfluence3000Mu', 'alpha5PlayerInfluence4000Mu', 
'alpha5PlayerInfluence5000Mu', 'alpha5PlayerInfluence6000Mu', 'alpha5PlayerInfluence7000Mu',
'alpha5PlayerInfluence8000Mu', 'alpha5PlayerInfluence9000Mu', 'alpha5PlayerInfluence10000Mu',
'alpha5PlayerInfluence2000Sigma', 'alpha5PlayerInfluence3000Sigma', 'alpha5PlayerInfluence4000Sigma',
'alpha5PlayerInfluence5000Sigma', 'alpha5PlayerInfluence6000Sigma', 'alpha5PlayerInfluence7000Sigma', 
'alpha5PlayerInfluence8000Sigma', 'alpha5PlayerInfluence9000Sigma', 'alpha5PlayerInfluence10000Sigma',   
'alpha1PlayerInfluence2000Max', 'alpha1PlayerInfluence3000Max', 'alpha1PlayerInfluence4000Max',
'alpha1PlayerInfluence5000Max', 'alpha1PlayerInfluence6000Max', 'alpha1PlayerInfluence7000Max',
'alpha1PlayerInfluence8000Max', 'alpha1PlayerInfluence9000Max', 'alpha1PlayerInfluence10000Max',
'alpha1PlayerInfluence2000Min', 'alpha1PlayerInfluence3000Min', 'alpha1PlayerInfluence4000Min',
'alpha1PlayerInfluence5000Min', 'alpha1PlayerInfluence6000Min', 'alpha1PlayerInfluence7000Min',
'alpha1PlayerInfluence8000Min', 'alpha1PlayerInfluence9000Min', 'alpha1PlayerInfluence10000Min', 
'alpha1PlayerInfluence2000Mu', 'alpha1PlayerInfluence3000Mu', 'alpha1PlayerInfluence4000Mu', 
'alpha1PlayerInfluence5000Mu', 'alpha1PlayerInfluence6000Mu', 'alpha1PlayerInfluence7000Mu',
'alpha1PlayerInfluence8000Mu', 'alpha1PlayerInfluence9000Mu', 'alpha1PlayerInfluence10000Mu', 
'alpha1PlayerInfluence2000Sigma', 'alpha1PlayerInfluence3000Sigma', 'alpha1PlayerInfluence4000Sigma',
'alpha1PlayerInfluence5000Sigma', 'alpha1PlayerInfluence6000Sigma', 'alpha1PlayerInfluence7000Sigma',
'alpha1PlayerInfluence8000Sigma', 'alpha1PlayerInfluence9000Sigma', 'alpha1PlayerInfluence10000Sigma']


num = list(np.unique(np.array(num)))
cat = list(np.unique(np.array(cat)))

# .011761 Model Input I Features
X1F = ['SHOTGUN_Formation','I_FORM_Formation','SINGLEBACK_Formation','PISTOL_Formation','back_moving_down_field',
       'back_from_scrimmage', 'min_dist', 'max_dist', 'mean_dist', 'std_dist', 'def_min_dist',
'def_max_dist','Orientation','def_mean_dist', 'def_std_dist','X', 'YREF', 'S', 'A', 'Dis', 'YardLine','fe5','fe8',
'fe10', 'fe11','PlayerAge','PlayerHeight_dense','DefendersInTheBox','TimeDelta','old_data','DirABS',
'OrientationABS','Dir_sin', 'Dir_cos','alpha7PlayerInfluence2000Max','premierPlayerInfluence5000Max',
'premierPlayerInfluence6000Max','premierPlayerInfluence7000Max',
'premierRusherX2000RushX', 
'premierRusherX4000RushX', 
'premierRusherX6000RushX', 'premierRusherX7000RushX',
'premierRusherX8000RushX',
'premierRusherX10000RushX',
'premierPlayerInfluence4000Min', 'premierPlayerInfluence6000Min',
'premierPlayerInfluence8000Min', 'premierPlayerInfluence10000Min',
'alpha7PlayerInfluence2000Mu','premierPlayerInfluence4000Mu',
'premierPlayerInfluence6000Mu','premierPlayerInfluence7000Mu','premierPlayerInfluence9000Mu',
'premierPlayerInfluence10000Mu','alpha7PlayerInfluence2000Sigma',
'premierPlayerInfluence4000Sigma','premierPlayerInfluence6000Sigma',
'premierPlayerInfluence9000Sigma','premierPlayerInfluence10000Sigma']

# Model Input II Features
X2F = ['SMAP1', 'CloseDistance1', 'DefenderAssessing1', 'RusherAssessing1', 'RunnerHeaded1', 'RunnerThinking1', 'DefenderEngagement1','DefensiveInfluence1','DistanceEstimate1','TimeTillImpact1',
       'SMAP2', 'CloseDistance2', 'DefenderAssessing2', 'RusherAssessing2', 'RunnerHeaded2', 'RunnerThinking2', 'DefenderEngagement2','DefensiveInfluence2','DistanceEstimate2',
       'SMAP3', 'CloseDistance3', 'DefenderAssessing3', 'RusherAssessing3', 'RunnerHeaded3', 'RunnerThinking3', 'DefenderEngagement3','DistanceEstimate3',
       'SMAP4', 'CloseDistance4', 'DefenderAssessing4', 'RusherAssessing4', 'RunnerHeaded4', 'RunnerThinking4', 'DefenderEngagement4','DistanceEstimate4', 
       'SMAP5', 'CloseDistance5', 'DefenderAssessing5', 'RusherAssessing5', 'RunnerHeaded5', 'RunnerThinking5', 'DefenderEngagement5','DistanceEstimate5', 
       'SMAP6', 'CloseDistance6', 'DefenderAssessing6', 'RusherAssessing6', 'RunnerHeaded6', 'RunnerThinking6', 'DefenderEngagement6',
       'SMAP7', 'CloseDistance7', 'DefenderAssessing7', 'RusherAssessing7', 'RunnerHeaded7', 'RunnerThinking7', 'DefenderEngagement7',
       'SMAP8', 'CloseDistance8', 'DefenderAssessing8', 'RusherAssessing8', 'RunnerHeaded8', 'RunnerThinking8', 'DefenderEngagement8',
       'SMAP9', 'CloseDistance9', 'DefenderAssessing9', 'RusherAssessing9', 'RunnerHeaded9', 'RunnerThinking9', 'DefenderEngagement9', 
       'SMAP10', 'CloseDistance10', 'DefenderAssessing10', 'RusherAssessing10', 'RunnerHeaded10', 
       'SMAP11', 'CloseDistance11', 'DefenderAssessing11', 'RusherAssessing11', 'RunnerHeaded11']



def get_pytorch_features():
    return ['SHOTGUN_Formation','I_FORM_Formation','SINGLEBACK_Formation','PISTOL_Formation','back_moving_down_field','premierRusherX6000RushX', 'premierRusherX7000RushX',
            'CloseDistance1','CloseDistance2','CloseDistance3','CloseDistance4','CloseDistance5','CloseDistance6','CloseDistance7','CloseDistance8','CloseDistance9','CloseDistance10','CloseDistance11',
            'back_from_scrimmage', 'mean_dist', 'std_dist', 'def_min_dist','def_mean_dist', 'def_std_dist','X', 'S','A', 'Dis','YardLine','fe5','fe8','fe10', 'fe11','PlayerAge',
            'PlayerHeight_dense','DefendersInTheBox','TimeDelta','old_data','OrientationABS','Dir_sin', 'Dir_cos',
            'premierPlayerInfluence2000Max','premierPlayerInfluence3000Max', 'premierPlayerInfluence4000Max','premierPlayerInfluence5000Max','premierPlayerInfluence6000Max','premierPlayerInfluence7000Max',
            'premierPlayerInfluence2000Min', 'premierPlayerInfluence4000Min','premierPlayerInfluence5000Min', 'premierPlayerInfluence6000Min','premierPlayerInfluence7000Min',
             'premierPlayerInfluence8000Min', 'premierPlayerInfluence2000Mu','premierPlayerInfluence3000Mu','premierPlayerInfluence4000Mu','premierPlayerInfluence5000Mu','premierPlayerInfluence6000Mu','premierPlayerInfluence7000Mu','premierPlayerInfluence8000Mu',
            'premierPlayerInfluence2000Sigma','premierPlayerInfluence3000Sigma','premierPlayerInfluence4000Sigma','premierPlayerInfluence5000Sigma','premierPlayerInfluence6000Sigma','premierPlayerInfluence7000Sigma','premierPlayerInfluence8000Sigma']


py_feats = get_pytorch_features()


# filter variables
num = [var for var in num if var in X1F+X2F+py_feats]
cat = [var for var in cat if var in X1F+X2F+py_feats]

def getNFLY(df):
    y = np.zeros(shape=(df.shape[0], 199))
    for i, yard in enumerate(df['Yards'].values):
        y[i, yard+99:] = np.ones(shape=(1, 100-yard))
    return y


# setup training data
X, GROUP, yards, y_pytorch  = tr_sf.copy(), tr_sf['GameId'].copy(), tr_sf.Yards, getNFLY(tr_sf)
X = process_two(X)
X = X.fillna(0)

# setup target data
y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][99 + target] = 1
    
# create Standard Scaler Transformation object
scaler = StandardScaler().fit(X[num])
X[num] = scaler.transform(X[num])

# create dictionary for categorical features
le_dict = {}

# create encoding objects
for ca in cat:
    le_dict[ca] = LabelEncoder().fit(X[ca].apply(str))
    X[ca]  = le_dict[ca].transform(X[ca].apply(str))

# retain only important model information
X   = X[cat+num]


# Create Training Input Layer Datasets
X1, X2 = X[X1F], X[X2F]

# seed everything
seed_everything(1234)

# set up parameters important variables
oof_preds, py_feats = np.ones((X.shape[0], 199)), get_pytorch_features()
torch_epoch, torch_batch, torch_len  = 100, 1012, X[py_feats].shape[1]



losses, models,pymodels, crps_csv, s_time   = [], [], [], [], time.time()

# Setup constants
kf = GroupKFold(n_splits=5)

# predictive model
oof_preds = np.ones((X.shape[0], 199))
pyt_oof = np.ones((X.shape[0], 199))
kyr_oof = np.ones((X.shape[0], 199))

for k_fold, (tix, vix) in enumerate(kf.split(X, y, GROUP)):
    
    # Pytorch Model Sigmoid
    # ------------------------
    
    # setup pytorch data
    tr_ds, val_df, tr_load, val_load = generate_dataloader(X[py_feats], y_pytorch, torch_batch, tix, vix)
    
    # train pytorch model
    pytorch_model = train_pytorch_model(tr_ds,val_df, tr_load, val_load,torch_len,torch_epoch,torch_batch,k_fold)
    
    # Keras Model Softmax
    # ------------------------
    
    # obtain model
    model1, crps_v1, crps_t1 = get_model([X1.ix[tix],X2.ix[tix]], y[tix], [X1.ix[vix],X2.ix[vix]], y[vix])
    model2, crps_v2, crps_t2 = get_model([X1.ix[tix],X2.ix[tix]], y[tix], [X1.ix[vix],X2.ix[vix]], y[vix])
    
    # get data out here
    if crps_v1 < crps_v2: model, crps_v, crps_t = model1, crps_v1, crps_t1
    else:                 model, crps_v, crps_t = model2, crps_v2, crps_t2
    
    # Upload Model Results
    # ------------------------
    
    # get model scores
    pyt_oof[vix] = pytorch_transform(model_eval(pytorch_model, val_df, val_load, torch_batch))*.15
    kyr_oof[vix] = keras_transform(model.predict([X1.ix[vix],X2.ix[vix]]))*.85
    
    # Run pytorch prediction
    oof_preds[vix] = pyt_oof[vix] + kyr_oof[vix]
    score = get_ypred_score(oof_preds[vix],y_pytorch[vix])
    print('Fold Dual Models CRPS: {:.6f}'.format(score)) 
    
    # print results
    plt.figure(figsize=(12, 5))
    plt.plot(model1.history.history['CRPS_score_val'])
    plt.plot(model2.history.history['CRPS_score_val'])
    plt.title("the %d fold crps validation is %f test is %f "%((k_fold+1),crps_v,crps_t)); plt.ylabel('CRPS'); plt.xlabel('epoch')
    plt.legend(['model 1: '+str(crps_v1),'model 2: '+str(crps_v2)], loc='upper right')
    plt.show()
    
    # append model & valuation
    pymodels.append(pytorch_model)
    models.append(model)
    crps_csv.append(crps_v) 
    del model1, model2, model
    gc.collect()
    
# predict mean crps score
print("mean crps is %f"%np.mean(crps_csv))
print('DONE OOF ALL CRPS: {:.5f}'.format(np.sum(np.power(oof_preds - y_pytorch, 2))/(199*len(oof_preds))))

def predict(x_te):
    model_num = len(models)
    for k,m in enumerate(models):
        if k==0: y_pred =  m.predict(x_te,batch_size=1024)
        else:    y_pred += m.predict(x_te,batch_size=1024)
            
    y_pred = y_pred / model_num
    return y_pred

def prediction_adjustment(y,yards):
    max_yards = 100-yards
    y[0,max_yards+99:] = np.ones(shape=(1, 100-max_yards))
    y[0,:yards]        = np.zeros(shape=(1, yards))
    return y

# ran with %%time in the Kaggle kernel
for (test_df, sample_prediction_df) in iter_test:
    basetable = create_features(test_df, deploy=True)
    X         = basetable.copy()
    X         = process_two(X)
    X         = X.fillna(0)
    
    X[num] = scaler.transform(X[num])
    
    for ca in cat:
        X[ca]  = le_dict[ca].transform(X[ca].apply(str))
    
    # predict keras transformation
    y_pred = keras_transform(predict([X[X1F], X[X2F]]))
    
    
    test_pyt  = TensorDataset(torch.from_numpy(X[py_feats].values))
    test_load = DataLoader(test_pyt, torch_batch, shuffle=False)

    mdl_n = len(pymodels)
    for k,m in enumerate(pymodels):
        if k==0: pyt_pred =  model_eval(m, test_load, test_load, torch_batch)/mdl_n
        else:    pyt_pred += model_eval(m, test_load, test_load, torch_batch)/mdl_n
    
    # pytorch transformation
    pyt_pre_new = pytorch_transform(pyt_pred)
    
    # create prediction data
    adjust_nfl_new  = pytorch_transform(pyt_pre_new*.15 + y_pred*.85)
    adjust_nfl_pred = prediction_adjustment(adjust_nfl_new,int(basetable.YardLine.values[0]))
    
    preds_df = pd.DataFrame(data=adjust_nfl_pred[0].reshape(1, 199), columns=sample_prediction_df.columns)
    env.predict(preds_df)

env.write_submission_file()

