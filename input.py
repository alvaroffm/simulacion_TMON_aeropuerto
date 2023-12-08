import pandas as pd
import seaborn as sns
import scipy as sc
from seaborn import kdeplot
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from miscelanea import dotdict
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def tabla_lambdas(tramos=14,plot=True):

    ################### VARIABLES##########
    archivo_excel='data.xlsx'
    ######################################


    # Lectura de excel y creación de Dataframe
    df_raw=pd.read_excel(archivo_excel)
    df=df_raw[['TIME','PAX']]
    df1=df.copy()
    df_tramos=df.copy()
    df1.loc[:, 'TIME'] = df['TIME'] - min(df['TIME'])
    n_pax=sum(df['PAX'])
    bins = np.linspace(int(min(df['TIME'])),max(df['TIME']),tramos+1)

    #Se divide el intervalo de operacion en tramos y se asigna un lambda
    df_tramos.loc[:, 'TRAMOS'] = pd.cut(df['TIME'], bins, include_lowest=True)
    newdf = df_tramos[['TRAMOS','PAX']].groupby('TRAMOS').sum()

    indices = newdf.index.to_series()


    df = newdf.reset_index()
    df['inicio'] = df['TRAMOS'].apply(lambda x: x.left).astype(float)
    hora_apertura = min(df['inicio'])
    df['inicio'] = (df['inicio']-hora_apertura)*60
    print(min(df['inicio']))
    df['fin'] = df['TRAMOS'].apply(lambda x: x.right).astype(float)
    df['fin'] = (df['fin'] - hora_apertura)*60

    df['lambda'] =(df['fin']-df['inicio'])/(1*df['PAX'])
    # df.replace([np.inf, -np.inf], 5, inplace=True)

    if plot:
        colorines = dotdict(dict(
            blue='#1A2732',
            green='#96CE00',
            blue_light='#3793FF'))

        figure1, ax = plt.subplots(1)
        df.plot(y='PAX', kind='bar', ax=ax, legend=False, color=colorines.blue_light, edgecolor=colorines.blue)

        # Set labels and title
        ax.set_xlabel('Hora')

        ax.set_ylabel('PAX')
        ax.set_title('PAX esperados')

        plt.savefig('images/prevision_aeropuerto.svg', dpi=300)

    return df




if __name__=='__main__':
    df=tabla_lambdas(14)