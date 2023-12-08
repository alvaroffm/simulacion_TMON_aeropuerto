# Paquetes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import random
from datetime import datetime
from matplotlib.ticker import MaxNLocator

# Importar modulos

from input import tabla_lambdas

def main():
    def aeropuerto(seed = 12345, freq = 30, time = 720, c = 5, Cola_max = 35, service_time = np.array([0.5, 1, 1.5]),
                   probabilities_nextS = [0.25, 0.6, 0.15]):

        """
        Simula el funcionamiento de un sistema de atención de pasajeros en un aeropuerto.

        Parámetros:
        - seed (int): Semilla para la generación de números aleatorios (por defecto 12345).
        - freq (int): Frecuencia para calcular la media de la cola (por defecto 30 minutos).
        - time (int): Tiempo total de simulación en minutos (por defecto 720 minutos).
        - c (int): Número inicial de servidores activos (por defecto 5).
        - Cola_max (int): Límite máximo de la cola antes de abrir nuevos servidores (por defecto 40).
        - service_time (numpy array): Tiempos de servicio para cada servidor (por defecto [0.5, 1, 1.5]).
        - probabilities_nextS (list): Probabilidades para elegir el próximo tiempo de servicio (por defecto [0.25, 0.6, 0.15]).

        Retorna:
        - data_aero (DataFrame): DataFrame que contiene la información de la simulación, con columnas como 't' (tiempo),
          'Cola' (número de pasajeros en cola), 'Llegadas' (número total de llegadas), 'Servicio i' (estado del i-ésimo servidor),
          'stay' (tiempo hasta el próximo evento), 'nextL' (próximo tiempo de llegada), 'nextS i' (próximo tiempo de servicio
          para el i-ésimo servidor) y 'Servidores activos' (número total de servidores activos en ese momento).
        """

        np.random.seed(seed)
        df_lambdas= tabla_lambdas(14,plot=False)
        cmax = c

        # Inicializaciones
        t = 0; Cola = 0; Servicio = [0] * c; Llegadas = 0; n = 1 # horas
        data_list = [] # lista temporal que guarda sobre el df

        # Obtener distribución de pasajeros por horas y lambdas
        filtered_df = df_lambdas[(df_lambdas['inicio'] <= t) & (t <= df_lambdas['fin'])]
        lambdaL = float(filtered_df['lambda'])
        nextL = np.random.exponential(scale=lambdaL) # Parámetro de llegadas markovianas

        # Obtener tiempos de servicio y parámetros del servicio
        initial_service_times = [np.random.choice(service_time, p=probabilities_nextS) for _ in range(c)] # Tiempos de servicio iniciales
        nextS = [nextL + initial_service_time for initial_service_time in initial_service_times] # Parámetro tiempo servicio

        stay = nextL # Inicializamos stay que nos servirá para almacenar el tiempo hasta el próximo evento

        # Inicialización del dataframe en función de c
        columns = ['t', 'Cola', 'Llegadas']
        columns += [f'Servicio {i+1}' for i in range(c)]
        columns += ['stay', 'nextL']
        columns += [f'nextS{i+1}' for i in range(c)]
        columns += [f'Servidores activos']
        data_aero = pd.DataFrame(columns={col: [] for col in columns})

        while min(nextL, min(nextS)) <= time: 
            t = nextL
            if nextL <= min(nextS): # Si la próxima llegada se produce antes que la próxima salida.
                if sum(Servicio) < c: # Si hay algún servidor libre.
                    for i in range(c):
                        if Servicio[i] == 0:
                            Servicio[i] += 1
                            nextS[i] = t + np.random.choice(service_time, p=probabilities_nextS)
                            break
                else:
                    Cola += 1 # Si no hay servidor libre aumentamos la cola.
                Llegadas += 1 # Aumentamos el contador de llegadas en 1.

                # for pt in range(1,len(pax_dist['t'])):
                    # if pax_dist['t'][pt-1] <= t and t < pax_dist['t'][pt]:
                    #     lambdaL = pax_dist['lambda_t'][pt-1]


                filtered_df = df_lambdas[(df_lambdas['inicio'] <= t) & (t <= df_lambdas['fin'])]
                lambdaL = float(filtered_df['lambda'])
                nextL = t + np.random.exponential(scale=lambdaL) # Calculamos la próxima llegada.

            else: # En otro caso, la salida en algún servidor se produce antes.
                t = min(nextS) 
                idx = nextS.index(t) # Encontramos el número de servidor libre.

                if Cola == 0: # Si no hay cola el servidor permanece vacío.
                    Servicio[idx] = 0
                    # nextS[c-1]=np.infty
                else: # Si hay cola, un pasajero de la cola ocupa el servidor.
                    Cola -= 1
                    Servicio[idx] = 1

                nextS[idx] = t + np.random.choice(service_time, p=probabilities_nextS)

            # Para cada hora, calculamos la media de la cola
            if t >= n*freq:
                data_hour = pd.DataFrame(data_list)
                data_hour.index = data_hour['t']
                data_hour['Cola media'] = data_hour['Cola'].rolling(window=freq).mean()
                if data_hour['Cola media'].iloc[-1] >= 5 and c < 4: # Si la media de la cola de la última hora
                    # es superior a 5 y no hay más de 4 servidores en activo, añadimos uno
                    c += 1 
                    Servicio[c-1] = 1
                    Cola -=1
                    nextS[c-1] = t + np.random.choice(service_time, p=probabilities_nextS)

                elif data_hour['Cola media'].iloc[-1] <= 1 and c > 1: # si la media de la cola es inferior a 1 y
                    # el número de servidores es mayor a 1

                    c -= 1 
                    Servicio[c-1] = 0
                n += 1

            stay = min(nextL, min(nextS)) - t # Almacenamos el tiempo hasta el próximo evento

            if Cola >= Cola_max:
                for i in range(c,cmax):
                    nextS[i] = t + np.random.choice(service_time, p=probabilities_nextS)
                    Servicio[i] = 1
                    Cola -=1

                c = cmax

                # Guardar cambios
            data_list.append({
            't': t,
            'Cola': Cola,
            'Llegadas': Llegadas,
            **{f'Servicio {i + 1}': Servicio[i] for i in range(c)},
            'stay': stay,
            'nextL': nextL,
            **{f'nextS{i + 1}': nextS[i] for i in range(c)}, 
            **{f'Servidores activos': c}
            })

        data_aero = pd.DataFrame(data_list, columns=columns)

        return data_aero, Cola_max, cmax, freq

    def metricas(df, Cola_max, freq):
        """
        Calcula diversas métricas a partir de los datos de una simulación de un sistema de atención de pasajeros en un aeropuerto.

        Parámetros:
        - df (DataFrame): DataFrame que contiene la información de la simulación, con columnas como 't' (tiempo),
          'Cola' (número de pasajeros en cola), 'Llegadas' (número total de llegadas), 'Servicio i' (estado del i-ésimo servidor),
          'stay' (tiempo hasta el próximo evento), 'nextL' (próximo tiempo de llegada), 'nextS i' (próximo tiempo de servicio
          para el i-ésimo servidor) y 'Servidores activos' (número total de servidores activos en ese momento).
        - Cola_max (int): parámetro que define el valor crítico de la cola a partir de la cual el sistema activa todos los servidores.
        - freq (int): Frecuencia para calcular la media de la cola.

        Retorna:
        - Wq (float): Tiempo medio de los clientes en cola.
        - Lq (float): Número medio de los clientes en cola.
        - L (float): Número medio de clientes en el sistema.
        - W (float): Tiempo medio de los clientes en el sistema.
        - Nh (DataFrame): Número medio de clientes por hora.
        - Ts (float): Número medio de servidores abiertos.
        - Ns (DataFrame): Tiempo de servidores abiertos para cada servidor.
        - llegadas_pax (float): Número total de llegadas.
        - TEPP_01 (float): Porcentaje de clientes en el sistema que están más de 10 minutos en cola.
        """

        c = max(df['Servidores activos'])
        cmax = c

        ## MÉTRICAS BÁSICAS

        # Tiempo medio de los clientes en cola (Wq)
        queue_time, wait_times = [], []
        for q in range(1,len(data_aero['Cola'])): 
            if data_aero['Cola'][q] > data_aero['Cola'][q-1]:
                queue_time.append(data_aero['t'][q])

            elif data_aero['Cola'][q] < data_aero['Cola'][q-1]:
                wait_times.append(data_aero['t'][q] - queue_time.pop(0))

        Wq = np.mean(wait_times)

        # Número medio de los clientes en cola (Lq)
        Lq = np.mean(data_aero['Cola'])

        # Número medio de clientes en el sistema (L)
        L = Lq + np.mean(data_aero['Servidores activos']) 

        # Tiempo medio de los clientes en el sistema (W)
        W = np.mean(abs(np.diff([np.mean(data_aero[f'nextS{i}']) for i in range(1, c+1)])))/freq + Wq 

        ## MÉTRICAS PARTICULARES DEL SISTEMA

        # Número medio de clientes por hora (Nh)

        data_aero['interval'] = (data_aero['t'] // 60)
        Nh = data_aero.groupby('interval').apply(lambda group: group['Servidores activos'].mean() + group['Cola'].mean()).reset_index()
        Nh.columns = ['interval', 'Nh']

        # Número medio de servidores abiertos (Ts)

        Ts = data_aero['Servidores activos'].mean()

        # Tiempo de servidores abiertos (Ns)

        Ns = []
        for i in range(1, cmax + 1):
            servidor_i_activo = data_aero[data_aero[f'Servicio {i}'].notna()]
            tiempo_servidor_i_activo = servidor_i_activo['stay'].sum()
            Ns.append({'Servidor': i, 'Tiempo_Activo': tiempo_servidor_i_activo})
        Ns = pd.DataFrame(Ns)

        # Número total de llegadas

        llegadas_pax = float(data_aero['Llegadas'].tail(1))

        # Porcentaje de clientes en el sistema que están más de 10 minutos en cola.

        Wq_out = 0
        for item in wait_times:
            if item > 10: Wq_out +=1

        TEPP_01 = 100 - (100 * Wq_out/llegadas_pax)

        return Wq, Lq, L, W, Nh, Ts, Ns, llegadas_pax, TEPP_01

    data_aero, Cola_max, cmax, freq = aeropuerto()

    # Resumen de la cola
    summary_cola = data_aero.groupby('Cola')['stay'].sum().reset_index()
    print(summary_cola)
    print((summary_cola['stay'] * summary_cola['Cola']).sum())

    # Resumen del servicio
    summary_servicio = data_aero.groupby('Llegadas')['stay'].sum().reset_index()
    print(summary_servicio)
    print((summary_servicio['stay'] * summary_servicio['Llegadas']).sum())

    Wq, Lq, L, W, Nh, Ts, Ns, llegadas_pax, TEPP_01 = metricas(df=data_aero, Cola_max = Cola_max, freq = freq)

    # Imprime las métricas
    print("Tiempo medio de los clientes en cola (Wq):", Wq)
    print("Número medio de los clientes en cola (Lq):", Lq)
    print("Número medio de clientes en el sistema (L):", L)
    print("Tiempo medio de los clientes en el sistema (W):", W)
    print("Número medio de clientes por hora (Nh):")
    print(Nh)
    print("Número medio de servidores abiertos (Ts):", Ts)
    print("Tiempo de servidores abiertos (Ns):")
    print(Ns)
    print("Número total de llegadas:", llegadas_pax)
    print("Porcentaje de clientes en el sistema que están más de 10 minutos en cola (TEPP_01):", TEPP_01)

    # Gráfico

    redimension_pax=float(data_aero['Llegadas'].tail(1))/100 #Este valor hace que la linea de llegadas represente los pasajeros acumulados durante el día (en porcentaje)

    fig,ax=plt.subplots(1,2,figsize=(15,6)) #Crea un segundo eje para representar el acumulado de las llegadas durante el día en porcentaje
    ax3 = ax[0].twinx()

    ax3.tick_params(axis='y', labelcolor='red')
    ax3.yaxis.label.set_color('red')
    ax3.tick_params(axis='y', colors='red')

    ax3.set_ylabel('% llegadas')
    ax3.plot(data_aero['t'], data_aero['Llegadas'] / redimension_pax, color='red', drawstyle='steps', label='Llegadas')
    ax3.legend(loc='upper right')


    ax[0].plot(data_aero['t'], data_aero['Cola'], color='purple', drawstyle='steps', label='Cola', linewidth=1)
    ax[0].text(15, Cola_max+0.5, f'{Cola_max}', fontsize=8, color='black',verticalalignment='center')
    ax[0].axhline(Cola_max,linestyle='dotted', linewidth=0.8, color='black')
    ax[0].set_xlabel('Tiempo (min)')
    ax[0].set_ylabel('Pasajeros en cola')
    ax[0].set_ylim(0,)
    ax[0].set_xlim(0,)
    ax[0].legend(loc='upper left')
    ax[0].set_title('Número de llegadas y pasajeros en cola', color='#890000')

    colores= ['#ff3333','#331100','#660000','#880000','#ff0000']*2
    ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    for i in range(1,cmax+1,1):
        data_aero[f'Servicio {i}'].replace(0, np.nan, inplace=True)
        ax[1].scatter(data_aero['t'],data_aero[f'Servicio {i}']+i-1,linewidths=0.2,color=colores[i], s=2,marker='x')

    # ax[1].scatter(data_aero['t'],(data_aero['Servidores activos']+0.1),s=10, color= 'green',label = "Serv. Abiertos")
    ax[1].plot(data_aero['t'],(data_aero['Servidores activos']+0.051),linewidth=1, color= 'black',label = "Serv. Abiertos")
    ax[1].legend()
    ax[1].set_ylabel('Servidores')
    ax[1].set_xlabel('Tiempo (min)')

    ax[1].set_title('Servidores abiertos y en uso',color='#890000')

    fig.tight_layout()
    
if __name__ == '__main__':
    main()