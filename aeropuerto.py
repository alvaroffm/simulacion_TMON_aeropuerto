def aeropuerto(seed = 12345, freq = 30, time = 720, c = 5, Cola_max = 40, service_time = np.array([0.5, 1, 1.5]),
               probabilities_nextS = [0.25, 0.6, 0.15]):
    
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
    
    return data_aero