#Video de Youtube explicando la Tarea

https://www.youtube.com/watch?v=WLZW9ml3kyA

# 📊 Utils.py - Módulo de Análisis Financiero

Un módulo completo de Python para análisis financiero que incluye extracción de datos, cálculo de métricas, gestión de carteras, simulaciones de Monte Carlo y visualización avanzada.

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Instalación](#-instalación)
- [Inicio Rápido](#-inicio-rápido)
- [Documentación](#-documentación)
  - [ExtractorFinanciero](#extractorfinanciero)
  - [SeleccionPrecios](#seleccionprecios)
  - [Operaciones](#operaciones)
  - [Cartera](#cartera)
  - [sim_mont](#sim_mont-simulación-de-monte-carlo)
  - [Graficos](#graficos)
- [Ejemplos Completos](#-ejemplos-completos)
- [Guía de Uso](#-guía-de-uso)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Características

- 🔄 **Extracción Multi-Proveedor**: Soporte para Yahoo Finance y Alpha Vantage
- 📈 **Análisis de Retornos**: Retornos logarítmicos, acumulados y volatilidad anualizada
- 💼 **Gestión de Carteras**: Creación y análisis de carteras con pesos personalizados
- 🎲 **Simulaciones Monte Carlo**: Simulación de trayectorias de precios con parámetros configurables
- 📊 **Visualización Avanzada**: Gráficos profesionales con seaborn y matplotlib
- 📄 **Reportes en Markdown**: Generación automática de reportes de carteras

---

## 🚀 Instalación

### Requisitos Previos

- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Instalación de Dependencias

```bash
# Opción 1: Desde requirements.txt
pip install -r requirements.txt

# Opción 2: Instalación manual
pip install yfinance alpha_vantage pandas numpy matplotlib seaborn openpyxl requests
```

### Verificar Instalación

```python
from Utils import ExtractorFinanciero
print("✅ Instalación correcta")
```

---

## 🎯 Inicio Rápido

```python
from Utils import ExtractorFinanciero, SeleccionPrecios, Operaciones, Cartera

# 1. Descargar datos
extractor = ExtractorFinanciero(proveedor='yfinance')
precios = extractor.obtener_precios(
    ticker=['AAPL', 'MSFT', 'GOOGL'],
    inicio='2020-01-01',
    fin='2024-12-31',
    cadencia='mensual'
)

# 2. Seleccionar precios de cierre
precios_close = SeleccionPrecios.seleccionar_precio(precios, 'Close')

# 3. Crear cartera equiponderada
cartera = Cartera(precios_close)
precios_cartera = cartera.calcular_precios_cartera()

# 4. Generar reporte
print(cartera.report())
```

---

## 📚 Documentación

### ExtractorFinanciero

Clase principal para extraer datos financieros de diferentes proveedores.

#### Inicialización

```python
from Utils import ExtractorFinanciero

# Yahoo Finance (sin API key)
extractor = ExtractorFinanciero(proveedor='yfinance')

# Alpha Vantage (requiere API key)
from config import ALPHA_API_KEY
extractor = ExtractorFinanciero(proveedor='alpha_vantage', api_key=ALPHA_API_KEY)
```

#### `obtener_precios(ticker, inicio, fin, cadencia, periodo)`

Descarga precios históricos OHLC (Open, High, Low, Close, Volume).

**Parámetros:**
- `ticker` (str o List[str]): Símbolo(s) de la acción
- `inicio` (str, opcional): Fecha inicio 'YYYY-MM-DD'
- `fin` (str, opcional): Fecha fin 'YYYY-MM-DD'
- `cadencia` (str): 'diario', 'semanal', 'mensual', 'trimestral', 'anual'
- `periodo` (str): Período para yfinance sin fechas ('1y', '6mo', etc.)

**Retorna:** DataFrame con columnas OHLC y Volume

**Ejemplos:**
```python
# Un solo ticker con fechas específicas
precios = extractor.obtener_precios(
    ticker='AAPL',
    inicio='2020-01-01',
    fin='2024-12-31',
    cadencia='mensual'
)

# Múltiples tickers
precios = extractor.obtener_precios(
    ticker=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    inicio='2020-01-01',
    fin='2024-12-31',
    cadencia='diario'
)

# Sin fechas específicas (usa período)
precios = extractor.obtener_precios(
    ticker='AAPL',
    periodo='1y',
    cadencia='diario'
)
```

#### `obtener_fundamentales(ticker, tipo, periodo)`

Descarga datos fundamentales (estados financieros).

**Parámetros:**
- `ticker` (str o List[str]): Símbolo(s) de la acción
- `tipo` (str): 'income_statement', 'balance_sheet', 'cash_flow', 'fundamentales'
- `periodo` (str): 'annual' o 'quarterly'

**Retorna:** Lista de objetos `DatosFundamentales` o Dict (para 'fundamentales')

**Ejemplo:**
```python
# Income statement anual
income_statements = extractor.obtener_fundamentales('AAPL', 'income_statement', 'annual')

# Datos fundamentales (retorna dict)
info = extractor.obtener_fundamentales('AAPL', 'fundamentales', 'annual')
print(info['marketCap'])
```

---

### SeleccionPrecios

Clase estática para seleccionar un tipo específico de precio del DataFrame OHLC.

#### `seleccionar_precio(df, precio)`

**Parámetros:**
- `df` (pd.DataFrame): DataFrame con columnas OHLC (puede ser MultiIndex para múltiples tickers)
- `precio` (str): 'Close', 'Open', 'High', 'Low' (por defecto 'Close')

**Retorna:** DataFrame con solo las columnas del precio seleccionado

**Ejemplo:**
```python
from Utils import SeleccionPrecios

# Seleccionar precios de cierre
precios_close = SeleccionPrecios.seleccionar_precio(precios, 'Close')

# Seleccionar precios de apertura
precios_open = SeleccionPrecios.seleccionar_precio(precios, 'Open')
```

**Nota:** Si el DataFrame tiene MultiIndex (múltiples tickers), el método maneja automáticamente la selección de 'Adj Close' cuando está disponible.

---

### Operaciones

Clase con métodos estáticos para realizar transformaciones financieras sobre DataFrames.

#### `retornos_logaritmicos(df)`

Calcula retornos logarítmicos usando operaciones matriciales vectorizadas.

**Parámetros:**
- `df` (pd.DataFrame): DataFrame con precios (índice datetime, columnas numéricas)

**Retorna:** DataFrame con retornos logarítmicos (primera fila con NaN eliminada)

**Fórmula:** `log(P_t) - log(P_{t-1}) = log(P_t / P_{t-1})`

**Ejemplo:**
```python
from Utils import Operaciones

retornos_log = Operaciones.retornos_logaritmicos(precios_close)
print(retornos_log.head())
```

#### `retornos_acumulados(df)`

Calcula retornos acumulados (performance acumulada) en base 100.

**Parámetros:**
- `df` (pd.DataFrame): DataFrame con precios (índice datetime, columnas numéricas)

**Retorna:** DataFrame con retornos acumulados (primer valor = 100.0 para cada columna)

**Fórmula:** `(P_t / P_0) * 100`

**Ejemplo:**
```python
acumulados = Operaciones.retornos_acumulados(precios_close)
print(acumulados.head())
# Primera fila siempre será 100.0
```

#### `volatilidad_anualizada(df, cadencia)`

Calcula la volatilidad anualizada usando retornos logarítmicos.

**Parámetros:**
- `df` (pd.DataFrame): DataFrame con precios
- `cadencia` (str): 'diario', 'semanal', 'mensual', 'trimestral', 'anual'

**Retorna:** Series con volatilidad anualizada para cada columna

**Fórmula:** `std(retornos_log) * sqrt(factor_anualizacion)`

**Factores de anualización:**
- Diario: √256
- Semanal: √52
- Mensual: √12
- Trimestral: √4
- Anual: √1

**Ejemplo:**
```python
volatilidad = Operaciones.volatilidad_anualizada(precios_close, cadencia='mensual')
print(volatilidad)
```

---

### Cartera

Clase para representar y analizar una cartera de activos con pesos personalizados o equiponderados.

#### Inicialización

```python
from Utils import Cartera

# Cartera equiponderada (todos los activos tienen el mismo peso)
cartera = Cartera(precios_close)

# Cartera con pesos personalizados (deben sumar 100)
cartera = Cartera(precios_close, pesos=[10, 15, 20, 15, 10, 10, 10, 5, 3, 2])

# Pesos como Series de pandas
pesos_series = pd.Series([10, 15, 20, 15, 10, 10, 10, 5, 3, 2], index=precios_close.columns)
cartera = Cartera(precios_close, pesos=pesos_series)

# Pesos como array numpy
pesos_array = np.array([10, 15, 20, 15, 10, 10, 10, 5, 3, 2])
cartera = Cartera(precios_close, pesos=pesos_array)
```

**Parámetros:**
- `precios` (pd.DataFrame): DataFrame con precios históricos (índice datetime, columnas = activos)
- `pesos` (opcional): List, pd.Series o np.ndarray con pesos en porcentaje (deben sumar 100 o 1.0)

**Validaciones:**
- Los pesos deben sumar 100 (porcentajes) o 1.0 (decimales, se convierten automáticamente)
- No se permiten pesos negativos
- El número de pesos debe coincidir con el número de activos

#### `calcular_precios_cartera()`

Calcula los precios de la cartera en base 100 usando retornos simples y encadenamiento.

**Método:**
1. Calcula retornos simples: `r_t = (P_t - P_{t-1}) / P_{t-1}`
2. Calcula retorno ponderado de la cartera: `r_cartera = Σ(w_i * r_i)`
3. Encadena retornos: `precio_t = precio_{t-1} * (1 + r_cartera)`
4. Normaliza a base 100: `precio_0 = 100.0`

**Retorna:** DataFrame con una columna 'Cartera' con precios en base 100

**Ejemplo:**
```python
precios_cartera = cartera.calcular_precios_cartera()
print(precios_cartera.head())
# Primera fila será 100.0
```

#### `report()`

Genera un reporte completo en formato Markdown con métricas clave de la cartera.

**Retorna:** Cadena de texto en formato Markdown

**Contenido del reporte:**
- Período de análisis (fechas inicio/fin, cadencia detectada)
- Retorno promedio anualizado (%) - calculado con retornos logarítmicos
- Retorno acumulado (%) - del primer al último período
- Volatilidad anualizada (%)
- Número de activos
- Número de períodos

**Ejemplo:**
```python
reporte = cartera.report()
print(reporte)

# Guardar en archivo
with open('reporte_cartera.md', 'w', encoding='utf-8') as f:
    f.write(reporte)
```

---

### sim_mont (Simulación de Monte Carlo)

Clase para realizar simulaciones de Monte Carlo usando el modelo de movimiento browniano geométrico.

#### Inicialización

```python
from Utils import sim_mont

simulacion = sim_mont(
    n_simulaciones=1000,
    horizonte=252,
    retorno_esperado=0.10,      # 10% anualizado
    volatilidad=0.20,            # 20% anualizado
    precio_inicial=100.0,
    cadencia='diario',
    semilla=42                   # Opcional para reproducibilidad
)
```

**Parámetros:**
- `n_simulaciones` (int): Número de simulaciones a realizar
- `horizonte` (int): Número de períodos a simular
- `retorno_esperado` (float, np.ndarray o pd.Series): Retorno esperado anualizado
  - Si es escalar: se usa para todos los períodos
  - Si es array/Series: debe tener tamaño = horizonte
- `volatilidad` (float, np.ndarray o pd.Series): Volatilidad anualizada
  - Si es escalar: se usa para todos los períodos
  - Si es array/Series: debe tener tamaño = horizonte
- `precio_inicial` (float): Precio inicial (por defecto 100.0)
- `cadencia` (str): 'diario', 'semanal', 'mensual', 'trimestral', 'anual'
- `semilla` (int, opcional): Semilla para reproducibilidad

**Modelo utilizado:**
- Retorno del período: `r_t = μ/periodos_año + (σ/√periodos_año) * ε_t`
- Precio: `P_t = P_0 * ∏(1 + r_i)` donde `i` va de 1 a `t`

#### `simular()`

Ejecuta la simulación de Monte Carlo.

**Retorna:** DataFrame con `n_simulaciones` columnas y `horizonte + 1` filas
- Columnas: `Sim_1`, `Sim_2`, ..., `Sim_n`
- Índice: `Periodo` de 0 a `horizonte`
- Primera fila (período 0): todas las simulaciones tienen valor `precio_inicial`

**Ejemplo:**
```python
resultados = simulacion.simular()
print(resultados.head())
print(f"Forma: {resultados.shape}")  # (horizonte+1, n_simulaciones)
```

#### `visualizar(df_simulacion, mostrar_todas, mostrar_percentiles, mostrar_media, percentiles, figsize, titulo)`

Visualiza los resultados de la simulación usando seaborn.

**Parámetros:**
- `df_simulacion` (pd.DataFrame, opcional): DataFrame con resultados (si None, ejecuta simular())
- `mostrar_todas` (bool): Si True, muestra trayectorias individuales (máx 100 si hay muchas)
- `mostrar_percentiles` (bool): Si True, muestra líneas de percentiles
- `mostrar_media` (bool): Si True, muestra la media de todas las simulaciones
- `percentiles` (List[float]): Percentiles a mostrar (por defecto [5, 25, 50, 75, 95])
- `figsize` (tuple): Tamaño de la figura (ancho, alto)
- `titulo` (str, opcional): Título personalizado

**Ejemplo:**
```python
# Visualización básica
simulacion.visualizar()

# Visualización personalizada
simulacion.visualizar(
    mostrar_todas=False,
    mostrar_percentiles=True,
    mostrar_media=True,
    percentiles=[10, 25, 50, 75, 90],
    figsize=(14, 8),
    titulo='Simulación de Precios - 10 años'
)
```

---

### Graficos

Clase con métodos estáticos para generar visualizaciones financieras profesionales.

#### `grafico_lineas(precios, figsize, titulo, mostrar_leyenda)`

Gráfico de líneas que muestra la trayectoria de precios de los activos.

**Parámetros:**
- `precios` (pd.DataFrame): DataFrame con precios históricos (índice datetime)
- `figsize` (tuple): Tamaño de la figura (ancho, alto), por defecto (12, 6)
- `titulo` (str, opcional): Título personalizado
- `mostrar_leyenda` (bool): Si True, muestra la leyenda (por defecto True)

**Ejemplo:**
```python
from Utils import Graficos

Graficos.grafico_lineas(precios_close, figsize=(14, 6))
```

#### `grafico_volatilidad_rentabilidad(df, col_volatilidad, col_rentabilidad, figsize, titulo, etiquetar_puntos)`

Scatterplot con volatilidad en el eje X y rentabilidad en el eje Y.

**Parámetros:**
- `df` (pd.DataFrame): DataFrame con columnas de volatilidad y rentabilidad
- `col_volatilidad` (str): Nombre de la columna de volatilidad
- `col_rentabilidad` (str): Nombre de la columna de rentabilidad
- `figsize` (tuple): Tamaño de la figura, por defecto (10, 6)
- `titulo` (str, opcional): Título personalizado
- `etiquetar_puntos` (bool): Si True, etiqueta cada punto con el índice del DataFrame

**Ejemplo:**
```python
# Crear DataFrame con métricas
df_metricas = pd.DataFrame({
    'Volatilidad (%)': [15, 20, 25],
    'Rentabilidad (%)': [10, 12, 15]
}, index=['AAPL', 'MSFT', 'GOOGL'])

Graficos.grafico_volatilidad_rentabilidad(
    df_metricas,
    col_volatilidad='Volatilidad (%)',
    col_rentabilidad='Rentabilidad (%)',
    etiquetar_puntos=True,
    titulo='Análisis Riesgo-Retorno'
)
```

#### `matriz_correlaciones(precios, figsize, titulo, annot, fmt, cmap, vmin, vmax)`

Genera una matriz de correlaciones con heatmap visual.

**Parámetros:**
- `precios` (pd.DataFrame): DataFrame con precios históricos (índice datetime)
- `figsize` (tuple): Tamaño de la figura, por defecto (10, 8)
- `titulo` (str, opcional): Título personalizado
- `annot` (bool): Si True, muestra los valores de correlación (por defecto False)
- `fmt` (str): Formato para los valores anotados (por defecto '.2f')
- `cmap` (str): Mapa de colores (por defecto 'coolwarm')
- `vmin`, `vmax` (float): Límites del mapa de colores (por defecto -1, 1)

**Retorna:** DataFrame con la matriz de correlaciones

**Nota:** Calcula correlaciones sobre retornos logarítmicos, no sobre precios.

**Ejemplo:**
```python
# Sin números (solo colores)
matriz = Graficos.matriz_correlaciones(precios_close)

# Con números
matriz = Graficos.matriz_correlaciones(precios_close, annot=True)

# Personalizado
matriz = Graficos.matriz_correlaciones(
    precios_close,
    figsize=(12, 10),
    titulo='Correlaciones entre Activos',
    annot=True,
    fmt='.3f',
    cmap='RdYlBu'
)
```

---

## 💡 Ejemplos Completos

### Ejemplo 1: Análisis Completo de Activos

```python
from Utils import ExtractorFinanciero, SeleccionPrecios, Operaciones, Graficos

# 1. Descargar datos
extractor = ExtractorFinanciero(proveedor='yfinance')
precios = extractor.obtener_precios(
    ticker=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
    inicio='2020-01-01',
    fin='2024-12-31',
    cadencia='mensual'
)

# 2. Seleccionar precios de cierre
precios_close = SeleccionPrecios.seleccionar_precio(precios, 'Close')

# 3. Calcular métricas
retornos_log = Operaciones.retornos_logaritmicos(precios_close)
retornos_acum = Operaciones.retornos_acumulados(precios_close)
volatilidad = Operaciones.volatilidad_anualizada(precios_close, cadencia='mensual')

# 4. Visualizar
Graficos.grafico_lineas(precios_close)
print(volatilidad)
```

### Ejemplo 2: Análisis de Cartera con Reporte

```python
from Utils import Cartera, Graficos

# Crear cartera con pesos personalizados
cartera = Cartera(precios_close, pesos=[25, 25, 20, 15, 15])

# Calcular precios de la cartera
precios_cartera = cartera.calcular_precios_cartera()

# Generar y mostrar reporte
reporte = cartera.report()
print(reporte)

# Visualizar trayectoria
Graficos.grafico_lineas(precios_cartera, titulo='Evolución de la Cartera')
```

### Ejemplo 3: Scatterplot Riesgo-Retorno

```python
import pandas as pd
from Utils import Operaciones, Graficos

# Calcular métricas para cada activo
retornos_log = Operaciones.retornos_logaritmicos(precios_close)
volatilidad = Operaciones.volatilidad_anualizada(precios_close, cadencia='mensual')

# Calcular rentabilidad promedio anualizada
factor_anual = 12  # Mensual
retorno_anual = retornos_log.mean() * factor_anual

# Crear DataFrame con métricas
df_metricas = pd.DataFrame({
    'Volatilidad (%)': volatilidad * 100,
    'Rentabilidad (%)': retorno_anual * 100
})

# Visualizar
Graficos.grafico_volatilidad_rentabilidad(
    df_metricas,
    col_volatilidad='Volatilidad (%)',
    col_rentabilidad='Rentabilidad (%)',
    etiquetar_puntos=True,
    titulo='Análisis Riesgo-Retorno por Activo'
)
```

### Ejemplo 4: Simulación de Monte Carlo Completa

```python
from Utils import sim_mont, Cartera, Operaciones

# 1. Obtener parámetros de una cartera real
cartera = Cartera(precios_close)
precios_cartera = cartera.calcular_precios_cartera()

# 2. Calcular retorno y volatilidad
retornos_log = Operaciones.retornos_logaritmicos(precios_cartera)
retorno_esperado = float(retornos_log.mean().iloc[0] * 12)  # Anualizar
volatilidad = float(Operaciones.volatilidad_anualizada(precios_cartera, cadencia='mensual').iloc[0])

print(f"Retorno esperado: {retorno_esperado*100:.2f}%")
print(f"Volatilidad: {volatilidad*100:.2f}%")

# 3. Crear simulación
simulacion = sim_mont(
    n_simulaciones=10000,
    horizonte=120,  # 10 años mensuales
    retorno_esperado=retorno_esperado,
    volatilidad=volatilidad,
    precio_inicial=100.0,
    cadencia='mensual',
    semilla=42
)

# 4. Ejecutar y visualizar
resultados = simulacion.simular()
simulacion.visualizar(
    mostrar_todas=True,
    mostrar_percentiles=True,
    mostrar_media=True,
    titulo='Simulación Monte Carlo - Cartera Equiponderada'
)
```

### Ejemplo 5: Matriz de Correlaciones

```python
from Utils import Graficos

# Matriz de correlaciones de los precios originales
matriz_original = Graficos.matriz_correlaciones(
    precios_close,
    titulo='Correlaciones entre Activos'
)

# Si tienes simulaciones (con índice datetime)
Simulador_series_con_fechas = Simulador_series.copy()
Simulador_series_con_fechas.index = pd.date_range(
    start='2025-01-01',
    periods=len(Simulador_series),
    freq='M'
)

matriz_simulacion = Graficos.matriz_correlaciones(
    Simulador_series_con_fechas,
    titulo='Correlaciones entre Simulaciones'
)
```

---

## 📖 Guía de Uso

### Flujo de Trabajo Típico

1. **Extracción de Datos**
   ```python
   extractor = ExtractorFinanciero(proveedor='yfinance')
   precios = extractor.obtener_precios(ticker=['AAPL', 'MSFT'], ...)
   precios_close = SeleccionPrecios.seleccionar_precio(precios, 'Close')
   ```

2. **Análisis de Activos Individuales**
   ```python
   retornos_log = Operaciones.retornos_logaritmicos(precios_close)
   volatilidad = Operaciones.volatilidad_anualizada(precios_close, cadencia='mensual')
   ```

3. **Creación de Cartera**
   ```python
   cartera = Cartera(precios_close, pesos=[...])
   precios_cartera = cartera.calcular_precios_cartera()
   reporte = cartera.report()
   ```

4. **Simulación**
   ```python
   simulacion = sim_mont(...)
   resultados = simulacion.simular()
   simulacion.visualizar()
   ```

5. **Visualización**
   ```python
   Graficos.grafico_lineas(precios_cartera)
   Graficos.matriz_correlaciones(precios_close)
   ```

### Buenas Prácticas

- ✅ **Siempre usa índices datetime** para los DataFrames de precios
- ✅ **Valida los pesos de la cartera** antes de crearla (deben sumar 100)
- ✅ **Especifica la cadencia correcta** para cálculos anualizados precisos
- ✅ **Usa semillas** en simulaciones Monte Carlo para reproducibilidad
- ✅ **Reinicia el kernel de Jupyter** después de modificar `Utils.py`

---

## ⚠️ Troubleshooting

### Error: "El índice del DataFrame debe ser datetime"

**Causa:** El DataFrame no tiene un índice de tipo datetime.

**Solución:**
```python
df.index = pd.to_datetime(df.index)
```

### Error: "Los pesos deben sumar 100"

**Causa:** Los pesos proporcionados no suman 100 (o 1.0 si son decimales).

**Solución:**
```python
# Verificar pesos
suma = sum(pesos)
print(f"Suma actual: {suma}")

# Ajustar si es necesario
pesos_normalizados = [p * 100 / suma for p in pesos]
```

### Error: "Alpha Vantage tiene un límite de 5 API calls"

**Causa:** Se están solicitando más de 5 tickers en el plan gratuito.

**Solución:**
- Limitar a 5 tickers por llamada
- Agregar pausas entre llamadas
- Considerar actualizar al plan de pago

### Error: "ValueError: El tamaño de retorno_esperado debe coincidir con horizonte"

**Causa:** Se está pasando un Series o array con tamaño diferente al horizonte.

**Solución:**
```python
# Asegurar que es un escalar (float)
retorno_esperado = float(retornos_log.mean().iloc[0] * 12)
volatilidad = float(volatilidad_series.iloc[0])
```

### Error: "AttributeError: 'DataFrame' object has no attribute 'pesos'"

**Causa:** Se está llamando a un método de instancia como método estático.

**Solución:**
```python
# ❌ Incorrecto
Cartera.calcular_precios_cartera(precios_close)

# ✅ Correcto
cartera = Cartera(precios_close)
precios_cartera = cartera.calcular_precios_cartera()
```

### Los tickers no se descargan

**Causas comunes:**
- Símbolos incorrectos para el proveedor
- Tickers internacionales con formato incorrecto
- Índices que requieren símbolos específicos

**Solución:**
- Verificar símbolos en Yahoo Finance o Alpha Vantage
- Probar con formatos alternativos (ej: 'GOOG' vs 'GOOGL')
- Consultar la documentación del proveedor

---

## 📝 Notas Importantes

### Límites de API

- **Yahoo Finance**: Sin límites oficiales, pero puede tener rate limiting
- **Alpha Vantage**: 
  - Plan gratuito: 5 API calls por minuto
  - El módulo incluye advertencias automáticas
  - Pausas de 200ms entre llamadas para múltiples tickers

### Formatos de Datos

- **Índice**: Siempre debe ser datetime
- **Columnas**: Deben ser numéricas (precios)
- **Pesos**: Pueden ser porcentajes (suman 100) o decimales (suman 1.0)

### Factores de Anualización

Los factores utilizados para anualizar métricas:

| Cadencia | Factor | Nota |
|----------|--------|------|
| Diario | 256 | Días hábiles |
| Semanal | 52 | Semanas por año |
| Mensual | 12 | Meses por año |
| Trimestral | 4 | Trimestres por año |
| Anual | 1 | Sin anualización |

### Orden de Columnas

**Nota:** Cuando se descargan múltiples tickers, las columnas pueden aparecer en orden alfabético en lugar del orden de la lista original. Esto es normal y no afecta los cálculos.

---

## 📄 Licencia

Este módulo está diseñado para uso educativo y profesional en análisis financiero.

---

## 👤 Autor

Desarrollado para el Curso Python Master - Tarea 1

---

## 🤝 Contribuciones

Para sugerencias o mejoras, por favor abre un issue o envía un pull request.

---

## 📞 Soporte

Para problemas o preguntas:
1. Revisa la sección [Troubleshooting](#-troubleshooting)
2. Verifica que todas las dependencias estén instaladas
3. Asegúrate de tener la versión más reciente de `Utils.py`

