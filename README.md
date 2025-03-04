# LABORATORIO-2

***Luz Marina Valderrama-5600741***

***Shesly Nicole Colorado - 5600756***

***Samuel Esteban Fonseca Luna - 5600808***

El siguiente codigo tiene como objetivo interpretar los conceptos de concolucion, reconocer la correlación como una operación entre señales y la transformada como herramienta de análisis en el dominio de la
frecuencia realizando diferentes ejercicios propuestos, como lo son la convolucion a mano y luego a traves de codigo utilizando como datos, la cedula y el codigo de estudiante, tambien se tomo una serie de datos extraidos de la base de datos physionet. Los datos utilizados pertenecen a un estudio de electromiografia correspondiente a (https://physionet.org/content/emgdb/1.0.0/); los archivos utilizados junto con el siguiente codigo explicado, estan en este repositorio.

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from scipy.signal import correlate
    from scipy.fft import fft, fftfreq
    from scipy.signal import welch
    
Se importan librerías esenciales para el procesamiento de señales:

numpy: Para manipulación de arreglos numéricos.
matplotlib.pyplot: Para generar gráficos.
scipy.stats: Para estadísticas y distribuciones.
scipy.signal: Para correlación y análisis de señales.
scipy.fft: Para la Transformada de Fourier.
welch: Para calcular la Densidad Espectral de Potencia.

# Definir las señales h[n] y x[n]

Se definen las señales h[n] y x[n] para cada integrante del grupo.
h[n] representa el código del estudiante y x[n] su número de cédula.

    hs = np.array([5, 6, 0, 0, 8, 0, 8])
    xs = np.array([1, 0, 5, 7, 9, 7, 9, 0, 9, 1])

    hl = np.array([5, 6, 0, 0, 7, 4, 1])
    xl = np.array([1, 0, 1, 3, 2, 6, 1, 1, 2, 1])

    hsh = np.array([5, 6, 0, 0, 7, 5, 6])
    xsh = np.array([1, 1, 0, 6, 6, 3, 3, 2, 7, 0])

Acá se definieron arreglos para la señal de entrada y su respuesta o salida con los códigos y cedulas de cada integrante del grupo.

# Realizar la convolución

Se calcula la convolución entre h[n] y x[n] para cada estudiante.
La convolución es una operación que representa cómo una señal se modifica al pasar por un sistema.


    ys = np.convolve(xs, hs, mode='full')
    yl = np.convolve(xl, hl, mode='full')
    ysh = np.convolve(xsh, hsh, mode='full')

# Graficar la señal resultante

    print("Señal Samuel y[n]:", ys)



Se muestran los resultados de la convolución gráficamente usando plt.stem(), que es útil para representar señales discretas.

    plt.stem(ys)
    plt.title('Gráfico de la señal Samuel  y[n]')
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.show()

![Image](https://github.com/user-attachments/assets/cc7ca018-4bb9-4fd4-b4f2-1b51321a3dae)

    print("Señal luz y[n]:", yl)
    plt.stem(yl)
    plt.title('Gráfico de la señal Luz y[n]')
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.show()

![Image](https://github.com/user-attachments/assets/01df60c3-6a22-41a7-b8c5-b6810ce7b07f)

    print("Señal shesly y[n]:", ysh)
    plt.stem(ysh)
    plt.title('Gráfico de la señal  Shesly y[n]')
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.show()

![Image](https://github.com/user-attachments/assets/3d34a2d0-16a3-4c0a-9bed-6140af9cca31)

Se realizo la convolución de cada sistema y se grafico su respectiva grafica de cada estudiante.

# Definir parámetros

Se establece la frecuencia de muestreo (Fs) y el período de muestreo (Ts),
definiendo el tiempo entre cada muestra.

     Fs = 1 / (1.25e-3)  # Frecuencia de muestreo inversa de Ts
     Ts = 1.25e-3  # Período de muestreo
     n = np.arange(0, 9)  # Valores de n

# Definir señales
     x1 = np.cos(2 * np.pi * 100 * n * Ts)
     x2 = np.sin(2 * np.pi * 100 * n * Ts)

Se definen las señales sin y cos para realizar la correlación entre estas con frecuencia de 100 Hz..

# Calcular la correlación

Se calcula la correlación cruzada entre x1[n] y x2[n],
lo que mide la similitud entre ambas señales en distintos desplazamientos (lags).

     correlacion = correlate(x1, x2, mode='full')
     lags = np.arange(-len(x1) + 1, len(x1))

# Graficar señales originales

Se representa gráficamente las señales originales x1[n] y x2[n]
y luego su correlación cruzada en un segundo gráfico.

     plt.figure(figsize=(12, 5))
     plt.subplot(2, 1, 1)
     plt.stem(n, x1, linefmt='b-', markerfmt='bo', basefmt='r-')
     plt.stem(n, x2, linefmt='g-', markerfmt='go', basefmt='r-')
     plt.title('Señales x1[n] y x2[n]')
     plt.xlabel('n')
     plt.ylabel('Amplitud')
     plt.legend(['x1[n]', 'x2[n]'])
     plt.grid()

## Graficar la correlación

     plt.subplot(2, 1, 2)
     plt.stem(lags, correlacion, linefmt='m-', markerfmt='mo', basefmt='r-')
     plt.title('Correlación entre x1[n] y x2[n]')
     plt.xlabel('Lags')
     plt.ylabel('Amplitud')
     plt.grid()
     plt.tight_layout()
     plt.show()

![Image](https://github.com/user-attachments/assets/5328fe80-7576-413e-aafa-c3ee5b90dd0e)

# Funciones para cargar datos


      def cargarDat(nombre):
    """Carga archivo .dat con la señal ECG"""
      with open(nombre, 'rb') as f:
        return np.fromfile(f, dtype=np.int16)

     def cargarHea(nombre):
    """Lee el archivo .hea y extrae parámetros de la señal"""
     with open(nombre, 'r') as f:
        lineas = f.readlines()

    fs = int(lineas[0].split()[2])  # Frecuencia de muestreo
    g = int(lineas[1].split()[2].split('/')[0])   # Ganancia
    base = g  # Línea base
    return fs, g, base

# Cargar archivos
     dat = "emg_healthy.dat"
     hea = "emg_healthy.hea"

     ecg = cargarDat(dat)  
     fs, g, base = cargarHea(hea)

El código define dos funciones para la carga de datos. cargarDat(nombre) abre un archivo .dat y extrae la señal EMG en formato binario, devolviendo un arreglo de valores enteros de 16 bits. cargarHea(nombre) lee el archivo .hea asociado, extrayendo información clave como la frecuencia de muestreo, la ganancia y la línea base de la señal.

# Convertir señal a mV y corregir línea base
     ecgMv = (ecg - base) / g
     t = np.arange(len(ecgMv)) / fs
     descriptive_stats = {
    "Media": np.mean(ecgMv),
    "Mediana": np.median(ecgMv),
    "Desviación estándar": np.std(ecgMv),
    "Mínimo": np.min(ecgMv),
    "Máximo": np.max(ecgMv)
    }
      print("Estadísticos descriptivos:", descriptive_stats)

      
Se corrige la señal restando la línea base y normalizándola con la ganancia.
También se calculan estadísticas básicas de la señal.



### Crear eje de tiempo completo
     tiempo_total = len(ecg) / fs
    t = np.linspace(0, tiempo_total, len(ecg))

### Parámetros del zoom
     tiempo_inicio = 0
     tiempo_fin = 0.1

### Índices del rango de tiempo deseado
    indice_inicio = int(tiempo_inicio * fs)
    indice_fin = int(tiempo_fin * fs)

### Recortar la señal y el tiempo
     t_zoom = t[indice_inicio:indice_fin]
     senal_zoom = ecg[indice_inicio:indice_fin]

# Zoom Fourier
    f_zoom = np.fft.fft(senal_zoom)
    freq_zoom = np.fft.fftfreq(len(senal_zoom), d=1/fs)

#### FFT de la señal
     ecgMv -= np.mean(ecgMv)
     N = len(ecgMv)
     fig, ax = plt.subplots(2, 1, figsize=(12, 6))
     freqs = np.fft.fftfreq(N, 1/fs)
     fft_values = np.fft.fft(ecgMv)
     psd = (np.abs(fft_values) ** 2) / N


# Frecuencia media y mediana
     freq_mean = np.sum(freqs * psd) / np.sum(psd)
     freq_median = freqs[np.argsort(psd)[len(psd)//2]]
     std_freq = np.std(freqs)

     print(f"Frecuencia media: {freq_mean:.2f} Hz")
     print(f"Frecuencia mediana: {freq_median:.2f} Hz")
     print(f"Desviación estándar de frecuencia: {std_freq:.2f} Hz")

     
# Graficar la señal en el tiempo con zoom
    plt.subplot(2, 1, 1)
    plt.plot(t_zoom, senal_zoom, color="blue", label="Señal EMG (Zoom)")
    plt.title("Señal EMG en el tiempo")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (mV)")
    plt.legend()
    plt.grid()

# Graficar la Transformada de Fourier (sólo frecuencias positivas)
    plt.subplot(2, 1, 2)
    plt.plot(freq_zoom[:len(freq_zoom)//2], abs(f_zoom[:len(f_zoom)//2])**2, color="red", label="FFT (Potencia)")
    plt.title("Transformada de Fourier de la señal EMG")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Potencia")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    reconstructed_signal = np.fft.ifft(fft_values)
    plt.plot(t, ecgMv, label="Señal Original")
    plt.plot(t, reconstructed_signal.real, label="Señal Reconstruida", linestyle="--")
    plt.legend()

#![Image](https://github.com/user-attachments/assets/33af4d77-1bac-42b9-9622-f145372bf752)

# Graficar Densidad Espectral de Potencia
     frequencies, psd = welch(ecgMv, fs, nperseg=1024)
     plt.figure(figsize=(10, 5))
     plt.semilogy(frequencies, psd, color="purple")  
     plt.xlabel("Frecuencia (Hz)")
     plt.ylabel("Densidad Espectral de Potencia (V²/Hz)")
     plt.title("Densidad Espectral de Potencia (PSD) de la señal EMG")
     plt.grid()
     plt.show()

Para obtener una mejor representación de la señal en frecuencia, se utiliza el método de Welch para Densidad Espectral de Potencia (PSD) .

![Image](https://github.com/user-attachments/assets/30d5c6ab-4508-4e88-999f-d5c341136e1d)

# Histograma de frecuencias en Frecuencias

    plt.figure(figsize=(8, 4))
    plt.hist(frequencies, bins=30, weights=psd, color='g', alpha=0.7)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Potencia acumulada')
    plt.title('Histograma de Potencia en Frecuencia')
    plt.grid()
    plt.show()

![Image](https://github.com/user-attachments/assets/b1381ce4-bdfa-4f95-beb6-4b99d2907ea8)

Muestra cómo se distribuyen las frecuencias en la señal y se usa bins=30para define itervalos en el histograma.

# Graficar señal EMG
     plt.figure(figsize=(20, 7))
     plt.plot(t, ecgMv, label="EMG (mV)", color="black")
     plt.xlabel("Tiempo (s)")
     plt.ylabel("Amp (mV)")
     plt.title("Señal EMG")
     plt.legend()
     plt.grid()
     plt.xlim([10, (t[-1] * 0.04) + 10])
     plt.show()

![Image](https://github.com/user-attachments/assets/6593101b-9fbb-4dd7-8ea8-b99d464853cc)

# Estadísticos con funciones
     media = np.mean(ecgMv)
     std = np.std(ecgMv)
     cv = std / media

Medios : Valor promedio de la seña
Desviación estándar : Medida de dispersión
Coeficiente de Variación (CV) 

# Histograma

     plt.figure(figsize=(10, 5))
     plt.hist(ecgMv, bins=50, color='blue', alpha=0.7, density=True, label="Histograma")
     xVals = np.linspace(min(ecgMv), max(ecgMv), 100)
    plt.plot(xVals, stats.norm.pdf(xVals, media, std), 'r-', label="Distribución Normal")
    plt.xlabel("Amp (mV)")
    plt.ylabel("Densidad")
    plt.title("Histograma de EMG")
    plt.legend()
    plt.grid()
    plt.show()
![image](https://github.com/user-attachments/assets/8b14a892-2b1f-4ece-847f-6ebe22e895a3)

Distribución de la amplitud de la señal en diferentes valores y se superpone una distribución normal para comparación.

# Función de Probabilidad Acumulada (CDF)

La CDF (Función de Distribución Acumulativa) permite ver cómo se distribuyen las amplitudes de la señal.

    ecgOrd = np.sort(ecgMv)
    cdf = np.arange(len(ecgOrd)) / len(ecgOrd)

    plt.figure(figsize=(10, 5))
    plt.plot(ecgOrd, cdf, label="CDF", color='green')
    plt.xlabel("Amp (mV)")
    plt.ylabel("Probabilidad Acumulada")
    plt.title("CDF de ECG")
    plt.legend()
    plt.grid()
    plt.show()

![Image](https://github.com/user-attachments/assets/b447419b-61ff-468c-98ab-b536d6b0d782)



# Mostrar resultados

Se imprimen los estadísticos descriptivos de la señal EMG utilizada.

    print("* Estadísticos de EMG:")
    print(f"* Media manual: {media:.4f} mV")
    print(f"* Desviación estándar: {std:.4f} mV")
    print(f"* Coeficiente de Variación : {cv:.4f}")

# Conclusiones


Se implementaron técnicas fundamentales de procesamiento de señales, como la convolución, correlación y Transformada de Fourier, lo que permitió analizar el comportamiento de la señal EMG tanto en el dominio del tiempo como en el dominio de la frecuencia.


La convolución entre las señales de entrada y los sistemas definidos con los códigos de los estudiantes permitió observar cómo se modifican las señales al pasar a través de un sistema. Esto es clave en el estudio de sistemas lineales en procesamiento digital de señales.


Se calculó la correlación cruzada entre dos señales sinusoidales (cos y sin) de la misma frecuencia. Se demostró que la correlación refleja el desfase entre ambas señales, confirmando que cos y sin son ortogonales.


Mediante la Transformada de Fourier se obtuvo el espectro de frecuencias de la señal EMG, lo que permitió analizar su composición en el dominio de la frecuencia. Se observó que la señal presenta componentes significativas en ciertas frecuencias, lo cual es útil para su clasificación y filtrado.


El uso del método de Welch para calcular la Densidad Espectral de Potencia (PSD) facilitó la identificación de las frecuencias con mayor contribución de energía en la señal EMG. Esta información es crucial para aplicaciones biomédicas, como la detección de actividad muscular o el diagnóstico de enfermedades neuromusculares...
