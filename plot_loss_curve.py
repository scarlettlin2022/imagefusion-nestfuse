import pandas as pd
import matplotlib.pyplot as plt
if __name__ == "__main__":
    train_loss_path='model/self-att_GM/fold0/2023-7-18-143544/train_0.xlsx'
    valid_loss_path='model/self-att_GM/fold0/2023-7-18-143544/valid_0.xlsx'
    
    train_df = pd.read_excel(train_loss_path)
    valid_df = pd.read_excel(valid_loss_path)
    #print(len(train_df))
    for i in range(len(train_df)):
        if (train_df.iloc[i].sum()-i)==0 and (valid_df.iloc[i].sum()-i)==0:
            
    
    
    plt.figure()
    plt.subplot(3,2,1)
    
    train_total = train_df['total'].to_numpy()
    valid_total = valid_df['total'].to_numpy()
    
    
    '''plt.plot(x,y1,label='train loss')
    plt.plot(x,y2,label='test loss')
    plt.title(title)
    plt.legend(loc='lower left')'''
    
    
    
    #print(train_total)
    #print(valid_total)