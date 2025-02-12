# LABORATORIO-2

***Luz Marina Valderrama-5600741***
***Shesly Nicole Colorado - 5600756***
***Samuel Esteban Fonseca Luna - 5600808***


    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from scipy.signal import correlate
    from scipy.fft import fft, fftfreq
    from scipy.signal import welch

# Definir las señales h[n] y x[n]

h[n] = Código
x[n] = Cédula

    hs = np.array([5, 6, 0, 0, 8, 0, 8])
    xs = np.array([1, 0, 5, 7, 9, 7, 9, 0, 9, 1])

    hl = np.array([5, 6, 0, 0, 7, 4, 1])
    xl = np.array([1, 0, 1, 3, 2, 6, 1, 1, 2, 1])

    hsh = np.array([5, 6, 0, 0, 7, 5, 6])
    xsh = np.array([1, 1, 0, 6, 6, 3, 3, 2, 7, 0])

Acá se definieron arreglos para la señal de entrada y su respuesta o salida con los códigos y cedulas de cada integrante del grupo.

# Realizar la convolución

    ys = np.convolve(xs, hs, mode='full')
    yl = np.convolve(xl, hl, mode='full')
    ysh = np.convolve(xsh, hsh, mode='full')

# Imprimir la señal resultante

    print("Señal Samuel y[n]:", ys)

# Graficar la señal resultante

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

     Fs = 1 / (1.25e-3)  # Frecuencia de muestreo inversa de Ts
     Ts = 1.25e-3  # Período de muestreo
     n = np.arange(0, 9)  # Valores de n

# Definir señales
     x1 = np.cos(2 * np.pi * 100 * n * Ts)
     x2 = np.sin(2 * np.pi * 100 * n * Ts)

Se definen las señales sin y cos para realizar la correlación entre estas.

# Calcular la correlación
     correlacion = correlate(x1, x2, mode='full')
     lags = np.arange(-len(x1) + 1, len(x1))

# Graficar señales originales
     plt.figure(figsize=(12, 5))
     plt.subplot(2, 1, 1)
     plt.stem(n, x1, linefmt='b-', markerfmt='bo', basefmt='r-')
     plt.stem(n, x2, linefmt='g-', markerfmt='go', basefmt='r-')
     plt.title('Señales x1[n] y x2[n]')
     plt.xlabel('n')
     plt.ylabel('Amplitud')
     plt.legend(['x1[n]', 'x2[n]'])
     plt.grid()

# Graficar la correlación

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

El código define dos funciones para la carga de datos. cargarDat(nombre) abre un archivo .dat y extrae la señal ECG en formato binario, devolviendo un arreglo de valores enteros de 16 bits. cargarHea(nombre) lee el archivo .hea asociado, extrayendo información clave como la frecuencia de muestreo, la ganancia y la línea base de la señal.

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

Se normaliza la señal y se extraen estadísticas descriptivas.


# Crear eje de tiempo completo
     tiempo_total = len(ecg) / fs
    t = np.linspace(0, tiempo_total, len(ecg))

# Parámetros del zoom
     tiempo_inicio = 0
     tiempo_fin = 0.1

# Índices del rango de tiempo deseado
    indice_inicio = int(tiempo_inicio * fs)
    indice_fin = int(tiempo_fin * fs)

# Recortar la señal y el tiempo
     t_zoom = t[indice_inicio:indice_fin]
     senal_zoom = ecg[indice_inicio:indice_fin]

#Zoom Fourier
    f_zoom = np.fft.fft(senal_zoom)
    freq_zoom = np.fft.fftfreq(len(senal_zoom), d=1/fs)

# FFT de la señal
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

# Graficar la señal en el rango seleccionado
     ax[0].plot(t_zoom, senal_zoom, color="blue", label="Señal EMG (Zoom)")
     ax[0].set_title(f"Señal EMG en el tiempo (de {tiempo_inicio}s a {tiempo_fin}s)")
     ax[0].set_xlabel("Tiempo (s)")
     ax[0].set_ylabel("Amplitud (mV)")
     ax[0].legend()
     ax[0].grid()


# Graficar la Transformada de Fourier (sólo frecuencias positivas)
     ax[1].plot(freq_zoom[:len(freq_zoom)//2], abs(f_zoom[:len(f_zoom)//2])**2, color="red", label="FFT (Potencia)")
     ax[1].set_title(f"Transformada de Fourier (Zoom: {tiempo_inicio}s - {tiempo_fin}s)")
     ax[1].set_xlabel("Frecuencia (Hz)")
     ax[1].set_ylabel("Potencia")
     ax[1].legend()
     ax[1].grid()

     plt.tight_layout()
     plt.show()

     reconstructed_signal = np.fft.ifft(fft_values)
     plt.plot(t, ecgMv, label="Señal Original")
     plt.plot(t, reconstructed_signal.real, label="Señal Reconstruida", linestyle="--")
     plt.legend()

![Image](https://github.com/user-attachments/assets/33af4d77-1bac-42b9-9622-f145372bf752)

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

# Graficar señal ECG
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
    plt.title("Histograma de ECG")
    plt.legend()
    plt.grid()
    plt.show()

![Image](https://github.com/user-attachments/assets/977ac973-0504-40f5-a1a9-13d2933218a3)

Distribución de la amplitud de la señal en diferentes valores y se superpone una distribución normal para comparación.

# Función de Probabilidad Acumulada (CDF)
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

La CDF (Función de Distribución Acumulativa) permite ver cómo se distribuyen las amplitudes de la señal.

# Mostrar resultados
    print("* Estadísticos de EMG:")
    print(f"* Media manual: {media:.4f} mV")
    print(f"* Desviación estándar: {std:.4f} mV")
    print(f"* Coeficiente de Variación : {cv:.4f}")
