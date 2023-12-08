def metricas(df = data_aero):
    
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
    W = np.mean(np.diff([np.mean(data_aero[f'nextS{i}']) for i in range(1, c+1)])) + Wq 
    
    ## MÉTRICAS PARTICULARES DEL SISTEMA
    
    # Número medio de clientes por hora (Nh)
    
    data_aero['interval'] = (data_aero['t'] // 60) * 60
    Nh = data_aero.groupby('interval').apply(lambda group: group['Servidores activos'].mean() + group['Cola'].mean()).reset_index()
    Nh.columns = ['interval', 'Nh']
    print(nh_per_interval)

    # Número medio de servidores abiertos (Ts)
    
    Ts = data_aero['Servidores activos'].mean()

    # Tiempo de servidores abiertos (Ns)
    
    Ns = []
    for i in range(1, cmax + 1):
        servidor_i_activo = data_aero[data_aero[f'Servicio {i}'] == 1]
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