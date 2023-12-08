# Simulación de un aeropuerto
Proyecto de la asignatura Técnicas de MonteCarlo basado en teoría de colas para la asignatura del mismo nombre en la Universidad Complutense de Madrid.
## Estructura de carpetas
### Carpeta raíz
- `main.py`: función principal a ejecutar. Realiza la simulación, obtiene las métricas y visualiza los datos de simulación.
- `aeropuerto.py`: método que ejecuta la simulación.
- `metricas.py`: método para extraer las métricas del sistema.
- `data.xlsx`: datos de donde se ha extraido la serie temporal sobre la que ejecutar la simulación.
- `input.py`: archivo que ajusta y crea la serie temporal por la cual se ejecuta la simulación.
- `miscelanea.py`: `options` para graficar.
- `requirements.txt`: fichero que contiene las dependencias necesarias para ejecutar la simulación.
### Otras carpetas
- `diagramas`: carpeta que contiene los diagramas de flujo y conceptuales del proyecto.
- `reuniones`: carpeta que contiene las reuniones realizadas mediante el proyecto.
- `bibliografía`: bibliografía externa consultada.
## Cómo ejecutar la simulación
### 1. Instalar ficheros.
En una git bash (en Windows) o en una bash o consola (en Linux o Mac) ejecutar el siguiente código en el directorio deseado:
`git clone https://github.com/alvaroffm/simulacion_TMON_aeropuerto.git`.
Una vez ejecutado el comando, se instalan los ficheros.
### 2. Instalar dependencias.
- En una consola tipo bash (Mac o Linux) o Powershell (Windows): Ejecutar el comando `pip install -r requirements.txt`. 
### 3. Ejecutar simulación.
En una consola tipo bash (Mac o Linux) o Powershell (Windows) con Python instalado, ejecutar el siguiente código en el directorio donde se han extraído los archivos:
`python main.py`.
Si el código se ejecuta sobre un entorno Jupyter, se puede ver más cómodamente las representaciones gráficas.

NOTA: Hay pocos commit porque este github se ha extraído de un entorno de pruebas: https://github.com/alvaroffm/TMON_filtro_aero.
