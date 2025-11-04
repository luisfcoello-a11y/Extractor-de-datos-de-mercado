import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import seaborn as sns
import matplotlib.pyplot as plt


class TipoDato(Enum):
    """Tipos de datos financieros disponibles"""
    PRECIOS = "precios"
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    FUNDAMENTALES = "fundamentales"


@dataclass
class DatosPrecios:
    """Representación estandarizada de datos de precios históricos"""
    ticker: str
    fecha_inicio: Optional[datetime] = None
    fecha_fin: Optional[datetime] = None
    cadencia: str = "diario"
    proveedor: str = "yfinance"
    datos: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    def __post_init__(self):
        """Validar que el DataFrame tenga la estructura correcta"""
        if not self.datos.empty:
            if not pd.api.types.is_datetime64_any_dtype(self.datos.index):
                raise ValueError("El índice debe ser datetime")
            if self.datos.index.name is None:
                self.datos.index.name = 'Date'
            # Asegurar que todas las columnas sean numéricas (precios)
            for col in self.datos.columns:
                if not pd.api.types.is_numeric_dtype(self.datos[col]):
                    raise ValueError(f"La columna '{col}' debe ser numérica")


@dataclass
class DatosFundamentales:
    """Representación estandarizada de datos fundamentales"""
    ticker: str
    tipo: TipoDato
    periodo: str  # 'annual' o 'quarterly'
    fecha_fiscal: datetime
    datos: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convertir los datos fundamentales a DataFrame"""
        if not self.datos:
            return pd.DataFrame()
        
        df = pd.DataFrame([self.datos], index=[self.fecha_fiscal])
        return df


class ExtractorFinanciero:
    """Clase para extraer datos financieros de diferentes proveedores"""
    
    def __init__(self, proveedor: str = 'yfinance', api_key: Optional[str] = None):
        """
        Inicializar el extractor
        
        Parámetros:
        - proveedor: 'yfinance' o 'alpha_vantage'
        - api_key: API key para Alpha Vantage (opcional, busca en config.py si no se proporciona)
        """
        self.proveedor = proveedor.lower()
        
        if self.proveedor not in ['yfinance', 'alpha_vantage']:
            raise ValueError(f"Proveedor '{proveedor}' no soportado. Use 'yfinance' o 'alpha_vantage'")
        
        # Cargar API key si es necesario
        if self.proveedor == 'alpha_vantage':
            if api_key:
                self.api_key = api_key
            else:
                from config import ALPHA_API_KEY
                self.api_key = ALPHA_API_KEY
        else:
            self.api_key = None
    
    def obtener_precios(self, ticker, inicio: Optional[str] = None, fin: Optional[str] = None,
                       cadencia: str = 'diario', periodo: str = '1y') -> pd.DataFrame:
        """
        Obtener precios históricos
        
        Parámetros:
        - ticker: símbolo o lista de símbolos
        - inicio: fecha de inicio 'YYYY-MM-DD' (opcional)
        - fin: fecha de fin 'YYYY-MM-DD' (opcional)
        - cadencia: 'diario', 'semanal', 'mensual', 'trimestral', 'anual'
        - periodo: para yfinance cuando no se especifican fechas
        
        Retorna:
        - DataFrame con Date como índice y columnas OHLC
        """
        # Validar cadencia
        cadencia = cadencia.lower()
        cadencias_validas = ['diario', 'semanal', 'mensual', 'trimestral', 'anual']
        if cadencia not in cadencias_validas:
            raise ValueError(f"Cadencia '{cadencia}' no válida. Use: {cadencias_validas}")
        
        # Convertir fechas
        fecha_inicio = pd.to_datetime(inicio) if inicio else None
        fecha_fin = pd.to_datetime(fin) if fin else None
        
        # Enrutar según el proveedor
        if self.proveedor == 'yfinance':
            datos_precios = self._extraer_precios_yfinance(ticker, fecha_inicio, fecha_fin, cadencia, periodo)
        elif self.proveedor == 'alpha_vantage':
            datos_precios = self._extraer_precios_alpha_vantage(ticker, fecha_inicio, fecha_fin, cadencia)
        
        return datos_precios.datos
    
    def obtener_fundamentales(self, ticker: Union[str, List[str]], tipo: str, periodo: str = 'annual') -> Union[List[DatosFundamentales], Dict]:
        """
        Obtener datos fundamentales
        
        Parámetros:
        - ticker: símbolo de la acción o lista de símbolos
        - tipo: 'income_statement', 'balance_sheet', 'cash_flow', 'fundamentales'
        - periodo: 'annual' o 'quarterly'
        
        Retorna:
        - Lista de objetos DatosFundamentales o Dict (para yfinance FUNDAMENTALES)
        """
        try:
            tipo_enum = TipoDato(tipo.lower())
        except ValueError:
            raise ValueError(f"Tipo de dato '{tipo}' no válida")
        
        # Enrutar según el proveedor
        if self.proveedor == 'yfinance':
            resultado = self._extraer_fundamentales_yfinance(ticker, tipo_enum, periodo)
        elif self.proveedor == 'alpha_vantage':
            resultado = self._extraer_fundamentales_alpha_vantage(ticker, tipo_enum, periodo)
        
        # Si es un diccionario (caso de yfinance FUNDAMENTALES), retornarlo tal cual
        if isinstance(resultado, dict):
            return resultado
        
        # Si es una lista, retornarla
        if isinstance(resultado, list):
            return resultado
        
        # Si es un objeto único, convertirlo a lista
        return [resultado]
    
    # ========== MÉTODOS PRIVADOS PARA YFINANCE ==========
    
    def _cadencia_a_intervalo_yfinance(self, cadencia: str) -> str:
        """Convertir cadencia a intervalo de yfinance"""
        cadencia = cadencia.lower()
        mapa = {
            'diario': '1d',
            'semanal': '1wk',
            'mensual': '1mo',
            'trimestral': '3mo',
            'anual': '1y'
        }
        return mapa.get(cadencia, '1d')
    
    def _extraer_precios_yfinance(self, ticker, inicio: Optional[datetime] = None,
                                  fin: Optional[datetime] = None, cadencia: str = "diario",
                                  periodo: str = '1y') -> DatosPrecios:
        """Extraer precios usando yfinance"""
        intervalo = self._cadencia_a_intervalo_yfinance(cadencia)
        
        # yfinance acepta listas de tickers
        if isinstance(ticker, list):
            tickers = ticker
        else:
            tickers = [ticker]
        
        # Descargar datos
        if inicio and fin:
            datos = yf.download(tickers, start=inicio.strftime('%Y-%m-%d'),
                               end=fin.strftime('%Y-%m-%d'), interval=intervalo, progress=False)
        else:
            datos = yf.download(tickers, period=periodo, interval=intervalo, progress=False)
        
        # Manejar MultiIndex: mantener todas las columnas OHLC
        if isinstance(datos.columns, pd.MultiIndex):
            # Si hay múltiples tickers, mantener la estructura MultiIndex
            # Si hay un solo ticker, simplificar las columnas
            if len(tickers) == 1:
                datos.columns = datos.columns.droplevel(1)
        
        # Normalizar índice
        if not pd.api.types.is_datetime64_any_dtype(datos.index):
            datos.index = pd.to_datetime(datos.index)
        datos.index = datos.index.astype('datetime64[ns]')
        datos.index.name = 'Date'
        
        # Crear objeto DatosPrecios
        if len(tickers) == 1:
            datos_precios = DatosPrecios(
                ticker=tickers[0],
                fecha_inicio=inicio,
                fecha_fin=fin,
                cadencia=cadencia,
                proveedor='yfinance',
                datos=datos
            )
        else:
            # Para múltiples tickers, crear un objeto combinado
            datos_precios = DatosPrecios(
                ticker=",".join(tickers),
                fecha_inicio=inicio,
                fecha_fin=fin,
                cadencia=cadencia,
                proveedor='yfinance',
                datos=datos
            )
        
        return datos_precios
    
    def _extraer_fundamentales_yfinance(self, ticker: str, tipo: TipoDato,
                                        periodo: str = 'annual') -> Union[List[DatosFundamentales], Dict]:
        """Extraer datos fundamentales usando yfinance"""
        tick = yf.Ticker(ticker)
        
        if tipo == TipoDato.INCOME_STATEMENT:
            if periodo == 'annual':
                datos_raw = tick.financials
            else:
                datos_raw = tick.quarterly_financials
        elif tipo == TipoDato.BALANCE_SHEET:
            if periodo == 'annual':
                datos_raw = tick.balance_sheet
            else:
                datos_raw = tick.quarterly_balance_sheet
        elif tipo == TipoDato.CASH_FLOW:
            if periodo == 'annual':
                datos_raw = tick.cashflow
            else:
                datos_raw = tick.quarterly_cashflow
        elif tipo == TipoDato.FUNDAMENTALES:
            datos_raw = tick.info
            return datos_raw
        else:
            raise ValueError(f"Tipo de dato no soportado: {tipo}")
        
        # Convertir a formato estándar
        if datos_raw is not None and not datos_raw.empty:
            resultados = []
            for fecha in datos_raw.columns:
                datos_fund = DatosFundamentales(
                    ticker=ticker,
                    tipo=tipo,
                    periodo=periodo,
                    fecha_fiscal=pd.to_datetime(fecha),
                    datos=datos_raw[fecha].to_dict()
                )
                resultados.append(datos_fund)
            
            return resultados
        else:
            raise ValueError("No se encontraron datos")
    
    # ========== MÉTODOS PRIVADOS PARA ALPHA VANTAGE ==========
    
    def _cadencia_a_metodo_alpha_vantage(self, cadencia: str):
        """Obtener el método de Alpha Vantage según la cadencia"""
        try:
            from alpha_vantage.timeseries import TimeSeries
        except ImportError:
            raise ImportError("Se requiere alpha_vantage. Instálalo con: pip install alpha_vantage")
        
        ts = TimeSeries(key=self.api_key, output_format='pandas')
        
        cadencia = cadencia.lower()
        
        # Alpha Vantage no soporta trimestral ni anual
        if cadencia in ['trimestral', 'anual']:
            raise ValueError(
                f"Alpha Vantage no soporta la cadencia '{cadencia}'. "
                "Cadencias disponibles: 'diario', 'semanal', 'mensual'"
            )
        
        mapa = {
            'diario': ts.get_daily,
            'semanal': ts.get_weekly,
            'mensual': ts.get_monthly
        }
        return mapa.get(cadencia, ts.get_daily)
    
    def _extraer_precios_alpha_vantage(self, ticker, inicio: Optional[datetime] = None,
                                       fin: Optional[datetime] = None, cadencia: str = "diario") -> DatosPrecios:
        """Extraer precios usando Alpha Vantage (soporta múltiples tickers)"""
        try:
            from alpha_vantage.timeseries import TimeSeries
        except ImportError:
            raise ImportError("Se requiere alpha_vantage. Instálalo con: pip install alpha_vantage")
        
        # Convertir a lista si es string
        if isinstance(ticker, str):
            tickers = [ticker]
        else:
            tickers = ticker
        
        # Advertencia si se solicitan más de 5 tickers
        if len(tickers) > 5:
            import warnings
            warnings.warn(
                f"⚠️ Advertencia: Estás solicitando {len(tickers)} tickers. "
                "Alpha Vantage tiene un límite de 5 API calls por minuto en el plan gratuito. "
                "Si necesitas más tickers, considera actualizar a la versión de pago para evitar agotar tus queries gratuitas.",
                UserWarning
            )
        
        metodo = self._cadencia_a_metodo_alpha_vantage(cadencia)
        
        # Lista para almacenar DataFrames de cada ticker
        dfs_tickers = []
        tickers_extraidos = []
        
        # Iterar sobre cada ticker y extraer sus datos
        for tck in tickers:
            try:
                datos, _ = metodo(symbol=tck, outputsize='full')
                
                # Normalizar índice
                if not pd.api.types.is_datetime64_any_dtype(datos.index):
                    columna_fecha = None
                    for col in ['Date', 'date', 'Fecha', 'fiscalDateEnding']:
                        if col in datos.columns:
                            columna_fecha = col
                            break
                    
                    if columna_fecha:
                        datos.index = pd.to_datetime(datos[columna_fecha])
                        datos = datos.drop(columns=[columna_fecha])
                
                datos.index = pd.to_datetime(datos.index).astype('datetime64[ns]')
                datos.index.name = 'Date'
                
                # Renombrar columnas de Alpha Vantage
                renombres = {
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. volume': 'Volume',
                    '1. Open': 'Open',
                    '2. High': 'High',
                    '3. Low': 'Low',
                    '4. Close': 'Close',
                    '5. Volume': 'Volume'
                }
                
                # Crear diccionario de renombres solo para columnas que existen
                renombres_aplicar = {k: v for k, v in renombres.items() if k in datos.columns}
                datos.rename(columns=renombres_aplicar, inplace=True)
                
                # Crear MultiIndex para las columnas (similar a yfinance con múltiples tickers)
                # Solo si hay múltiples tickers, de lo contrario mantener columnas simples
                if len(tickers) > 1:
                    datos.columns = pd.MultiIndex.from_tuples([
                        (col, tck) for col in datos.columns
                    ])
                
                dfs_tickers.append(datos)
                tickers_extraidos.append(tck)
                
                # Pequeña pausa para evitar exceder el límite de API calls (si hay múltiples tickers)
                if len(tickers) > 1:
                    import time
                    time.sleep(0.2)  # Pausa de 200ms entre llamadas
                    
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Error al extraer datos para {tck}: {str(e)}. Continuando con los demás tickers.",
                    UserWarning
                )
                continue
        
        if not dfs_tickers:
            raise ValueError("No se pudo extraer datos para ningún ticker")
        
        # Combinar todos los DataFrames usando join (operación matricial)
        # Empezar con el primer DataFrame
        datos_combinados = dfs_tickers[0]
        
        # Hacer join con el resto de DataFrames
        for df in dfs_tickers[1:]:
            datos_combinados = datos_combinados.join(df, how='outer')
        
        # Ordenar por fecha
        datos_combinados = datos_combinados.sort_index()
        
        # Si solo hay un ticker, simplificar el MultiIndex
        if len(tickers_extraidos) == 1:
            datos_combinados.columns = datos_combinados.columns.droplevel(1)
        
        datos_precios = DatosPrecios(
            ticker=",".join(tickers_extraidos) if len(tickers_extraidos) > 1 else tickers_extraidos[0],
            fecha_inicio=inicio,
            fecha_fin=fin,
            cadencia=cadencia,
            proveedor='alpha_vantage',
            datos=datos_combinados
        )
        
        return datos_precios
    
    def _extraer_fundamentales_alpha_vantage(self, ticker: Union[str, List[str]], tipo: TipoDato,
                                             periodo: str = 'annual') -> List[DatosFundamentales]:
        """Extraer datos fundamentales usando Alpha Vantage (soporta múltiples tickers)"""
        try:
            from alpha_vantage.fundamentaldata import FundamentalData
        except ImportError:
            raise ImportError("Se requiere alpha_vantage. Instálalo con: pip install alpha_vantage")
        
        fd = FundamentalData(key=self.api_key, output_format='pandas')
        
        # Convertir a lista si es string
        if isinstance(ticker, str):
            tickers = [ticker]
        else:
            tickers = ticker
        
        # Advertencia si se solicitan más de 5 tickers
        if len(tickers) > 5:
            import warnings
            warnings.warn(
                f"⚠️ Advertencia: Estás solicitando {len(tickers)} tickers. "
                "Alpha Vantage tiene un límite de 5 API calls por minuto en el plan gratuito. "
                "Si necesitas más tickers, considera actualizar a la versión de pago para evitar agotar tus queries gratuitas.",
                UserWarning
            )
        
        # Lista para almacenar todos los resultados
        todos_resultados = []
        
        # Iterar sobre cada ticker y extraer sus datos
        for tck in tickers:
            try:
                if tipo == TipoDato.INCOME_STATEMENT:
                    if periodo == 'annual':
                        datos, _ = fd.get_income_statement_annual(symbol=tck)
                    else:
                        datos, _ = fd.get_income_statement_quarterly(symbol=tck)
                elif tipo == TipoDato.BALANCE_SHEET:
                    if periodo == 'annual':
                        datos, _ = fd.get_balance_sheet_annual(symbol=tck)
                    else:
                        datos, _ = fd.get_balance_sheet_quarterly(symbol=tck)
                elif tipo == TipoDato.CASH_FLOW:
                    if periodo == 'annual':
                        datos, _ = fd.get_cash_flow_annual(symbol=tck)
                    else:
                        datos, _ = fd.get_cash_flow_quarterly(symbol=tck)
                else:
                    raise ValueError(f"Tipo de dato no soportado: {tipo}")
                
                # Convertir a formato estándar
                if datos is not None and not datos.empty:
                    for _, row in datos.iterrows():
                        fecha_fiscal = pd.to_datetime(row.get('fiscalDateEnding', row.name))
                        datos_fund = DatosFundamentales(
                            ticker=tck,
                            tipo=tipo,
                            periodo=periodo,
                            fecha_fiscal=fecha_fiscal,
                            datos=row.to_dict()
                        )
                        todos_resultados.append(datos_fund)
                
                # Pequeña pausa para evitar exceder el límite de API calls (si hay múltiples tickers)
                if len(tickers) > 1:
                    import time
                    time.sleep(0.2)  # Pausa de 200ms entre llamadas
                    
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Error al extraer datos para {tck}: {str(e)}. Continuando con los demás tickers.",
                    UserWarning
                )
                continue
        
        if not todos_resultados:
            raise ValueError("No se pudo extraer datos para ningún ticker")
        
        return todos_resultados
    
# ========== CLASE SELECCION PRECIOS ==========

class SeleccionPrecios:
    """Clase para seleccionar qué precio del OHLC usar"""
    
    @staticmethod
    def seleccionar_precio(df, precio='Close'):
        """
        Seleccionar un precio específico del DataFrame OHLC
        
        Parámetros:
        - df: DataFrame con columnas OHLC (Open, High, Low, Close, Volume, etc.)
               Puede tener columnas simples o MultiIndex (para múltiples tickers)
        - precio: 'Close', 'Open', 'High', 'Low' (por defecto 'Close')
        
        Retorna:
        - DataFrame con solo las columnas de precio seleccionado
        """
        if df.empty:
            raise ValueError("El DataFrame está vacío")
        
        # Validar que el índice sea datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise ValueError("El índice del DataFrame debe ser datetime")
        
        # Normalizar el nombre del precio
        precio = precio.capitalize()
        precios_validos = ['Close', 'Open', 'High', 'Low']
        
        if precio not in precios_validos:
            raise ValueError(f"Precio '{precio}' no válido. Use: {precios_validos}")
        
        # Verificar si las columnas son MultiIndex (múltiples tickers)
        if isinstance(df.columns, pd.MultiIndex):
            # Caso MultiIndex: buscar columnas en el primer nivel que coincidan con el precio
            columnas_seleccionadas = []
            
            # Si el precio es Close, buscar también 'Adj Close'
            if precio == 'Close':
                # Buscar 'Adj Close' primero (preferido para yfinance)
                for col in df.columns:
                    if col[0] == 'Adj Close':
                        columnas_seleccionadas.append(col)
                # Si no se encontró 'Adj Close', buscar 'Close'
                if not columnas_seleccionadas:
                    for col in df.columns:
                        if col[0] == precio:
                            columnas_seleccionadas.append(col)
            else:
                # Para Open, High, Low, buscar directamente
                for col in df.columns:
                    if col[0] == precio:
                        columnas_seleccionadas.append(col)
            
            if not columnas_seleccionadas:
                raise ValueError(f"No se encontró la columna '{precio}' en el DataFrame. Columnas disponibles en nivel 0: {df.columns.get_level_values(0).unique().tolist()}")
            
            # Retornar DataFrame con las columnas seleccionadas
            return df[columnas_seleccionadas]
        
        else:
            # Caso columnas simples: código original
            columna_encontrada = None
            
            # Si el precio es Close, buscar también 'Adj Close' (para yfinance)
            if precio == 'Close':
                if 'Adj Close' in df.columns:
                    columna_encontrada = 'Adj Close'
                elif precio in df.columns:
                    columna_encontrada = precio
            else:
                # Para Open, High, Low, buscar directamente
                if precio in df.columns:
                    columna_encontrada = precio
            
            if columna_encontrada is None:
                raise ValueError(f"No se encontró la columna '{precio}' en el DataFrame. Columnas disponibles: {list(df.columns)}")
            
            # Retornar DataFrame con solo la columna seleccionada (manteniendo el índice datetime)
            return pd.DataFrame({precio: df[columna_encontrada]}, index=df.index)


# ========== CLASE OPERACIONES (TRANSFORMACIONES SOBRE DATAFRAMES) ==========

class Operaciones:
    """Clase para realizar transformaciones sobre DataFrames financieros"""
    
    @staticmethod
    def retornos_logaritmicos(df):
        """
        Calcular retornos logarítmicos usando operaciones matriciales
        
        Parámetros:
        - df: DataFrame con índices datetime y columnas numéricas (precios)
        
        Retorna:
        - DataFrame con retornos logarítmicos
        """
        if df.empty:
            raise ValueError("El DataFrame está vacío")
        
        # Validar que el índice sea datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise ValueError("El índice del DataFrame debe ser datetime")
        
        # Seleccionar solo columnas numéricas
        df_numerico = df.select_dtypes(include=[np.number])
        
        if df_numerico.empty:
            raise ValueError("No se encontraron columnas numéricas en el DataFrame")
        
        # Calcular retornos logarítmicos usando operaciones matriciales
        # Aplicar log a todo el DataFrame, luego diff, todo vectorizado
        df_retornos = np.log(df_numerico).diff().dropna()
        
        
        return df_retornos
    
    @staticmethod
    def retornos_acumulados(df):
        """
        Calcular retornos acumulados (performance acumulada)
        
        Parámetros:
        - df: DataFrame con índices datetime y columnas numéricas (precios)
        
        Retorna:
        - DataFrame con retornos acumulados (base 100.0 en el primer período)
        """
        if df.empty:
            raise ValueError("El DataFrame está vacío")
        
        # Validar que el índice sea datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise ValueError("El índice del DataFrame debe ser datetime")
        
        # Seleccionar solo columnas numéricas
        df_numerico = df.select_dtypes(include=[np.number])
        
        if df_numerico.empty:
            raise ValueError("No se encontraron columnas numéricas en el DataFrame")
        
        # Calcular retornos acumulados usando divisiones con operaciones matriciales
        # Dividir cada columna por su primer valor (precio inicial) y multiplicar por 100
        # Esta operación es completamente vectorizada: cada fila se divide por la primera fila
        # La primera fila será 100.0 (porque dividimos por sí misma y multiplicamos por 100)
        df_acumulados = (df_numerico / df_numerico.iloc[0]) * 100
        
        return df_acumulados
    
    @staticmethod
    def volatilidad_anualizada(df, cadencia='diario'):
        """
        Calcular volatilidad anualizada
        
        Parámetros:
        - df: DataFrame con índices datetime y columnas numéricas (precios)
        - cadencia: 'diario', 'semanal', 'mensual', 'trimestral', 'anual'
        
        Retorna:
        - Series con volatilidad anualizada para cada columna
        """
        if df.empty:
            raise ValueError("El DataFrame está vacío")
        
        # Calcular retornos logarítmicos primero
        retornos = Operaciones.retornos_logaritmicos(df)
        
        # Factor de anualización según la cadencia
        cadencia = cadencia.lower()
        factores = {
            'diario': 256,
            'semanal': 52,
            'mensual': 12,
            'trimestral': 4,
            'anual': 1
        }
        
        factor = factores.get(cadencia, 256)
        
        # Calcular volatilidad anualizada (desviación estándar de retornos * sqrt(factor))
        volatilidad = retornos.std() * np.sqrt(factor)
        
        return volatilidad


class Cartera:
    """
    Clase para representar y trabajar con una cartera de activos
    
    Atributos:
    - precios: DataFrame con precios de los activos
    - pesos: Series con los pesos de cada activo en la cartera
    """
    
    def __init__(self, precios: pd.DataFrame, pesos: Optional[Union[pd.Series, np.ndarray, List]] = None):
        """
        Inicializar una cartera
        
        Parámetros:
        - precios: DataFrame con precios históricos (índice datetime, columnas = activos)
        - pesos: Opcional. Series de pandas, numpy array o lista con pesos de cada activo.
                 Si no se proporciona, se usan pesos equiponderados.
                 Los pesos deben sumar 100 (porcentajes) o 1.0 (decimales).
        """
        # Validar que precios sea un DataFrame no vacío
        if not isinstance(precios, pd.DataFrame):
            raise TypeError("precios debe ser un pandas DataFrame")
        
        if precios.empty:
            raise ValueError("El DataFrame de precios está vacío")
        
        # Validar que el índice sea datetime
        if not pd.api.types.is_datetime64_any_dtype(precios.index):
            raise ValueError("El índice del DataFrame debe ser datetime")
        
        # Validar que todas las columnas sean numéricas
        if not precios.select_dtypes(include=[np.number]).shape[1] == precios.shape[1]:
            raise ValueError("Todas las columnas del DataFrame deben ser numéricas")
        
        self.precios = precios
        
        # Obtener nombres de activos (columnas)
        activos = precios.columns.tolist()
        n_activos = len(activos)
        
        # Si no se proporcionan pesos, crear pesos equiponderados
        if pesos is None:
            # Pesos equiponderados: cada activo tiene peso 1/n_activos
            peso_equiponderado = 100.0 / n_activos
            self.pesos = pd.Series(
                [peso_equiponderado] * n_activos,
                index=activos,
                name='Peso (%)'
            )
        else:
            # Convertir pesos a Series si es necesario
            if isinstance(pesos, (list, np.ndarray)):
                if len(pesos) != n_activos:
                    raise ValueError(f"El número de pesos ({len(pesos)}) no coincide con el número de activos ({n_activos})")
                self.pesos = pd.Series(pesos, index=activos, name='Peso (%)')
            elif isinstance(pesos, pd.Series):
                # Si es Series, verificar que tenga el mismo número de elementos
                if len(pesos) != n_activos:
                    raise ValueError(f"El número de pesos ({len(pesos)}) no coincide con el número de activos ({n_activos})")
                # Si tiene índice, usar los valores pero mantener el orden de activos
                if pesos.index.equals(pd.Index(activos)):
                    self.pesos = pesos.copy()
                    self.pesos.name = 'Peso (%)'
                else:
                    # Si los índices no coinciden, usar solo los valores en el orden de activos
                    self.pesos = pd.Series(pesos.values, index=activos, name='Peso (%)')
            else:
                raise TypeError("pesos debe ser un pandas Series, numpy array, lista o None")
            
            # Validar que los pesos sumen 100 (con tolerancia de 0.001 para errores de redondeo)
            suma_pesos = self.pesos.sum()
            tolerancia = 0.001
            
            # Detectar si los pesos están en formato porcentual (suman ~100) o decimal (suman ~1.0)
            if abs(suma_pesos - 100.0) < tolerancia:
                # Ya están en porcentaje, no hacer nada
                pass
            elif abs(suma_pesos - 1.0) < tolerancia:
                # Están en formato decimal, convertir a porcentaje
                self.pesos = self.pesos * 100.0
            else:
                raise ValueError(
                    f"Los pesos deben sumar 100 (porcentajes) o 1.0 (decimales). "
                    f"Suma actual: {suma_pesos:.6f}"
                )
            
            # Validar que no haya pesos negativos (opcional, pero buena práctica)
            if (self.pesos < 0).any():
                raise ValueError("Los pesos no pueden ser negativos")
    
    def __repr__(self):
        """Representación de la cartera"""
        n_activos = len(self.pesos)
        suma_pesos = self.pesos.sum()
        return (
            f"Cartera(n_activos={n_activos}, "
            f"suma_pesos={suma_pesos:.2f}%, "
            f"periodo={self.precios.index[0].date()} a {self.precios.index[-1].date()})"
        )
    
    def calcular_precios_cartera(self) -> pd.DataFrame:
        """
        Calcular los precios de la cartera en base 100 usando operaciones lineales
        
        Calcula retornos simples, los multiplica por los pesos y encadena usando (1 + retorno)
        para obtener un índice base 100.
        
        Retorna:
        - DataFrame con una columna 'Cartera' que contiene los precios acumulados en base 100
        """
        # Convertir pesos a decimales para el cálculo
        pesos_decimal = self.pesos / 100.0
        
        # Calcular retornos simples: r_t = (P_t - P_{t-1}) / P_{t-1} = P_t / P_{t-1} - 1
        retornos = self.precios.pct_change()
        
        # El primer retorno será NaN (no hay precio anterior), lo rellenamos con 0
        retornos = retornos.fillna(0)
        
        # Calcular retorno de la cartera: suma ponderada de retornos
        # retorno_cartera_t = sum(peso_i * retorno_i_t)
        retorno_cartera = (retornos * pesos_decimal).sum(axis=1)
        
        # Encadenar retornos usando multiplicación: precio_t = precio_{t-1} * (1 + retorno_t)
        # Empezando desde 100: precio_0 = 100
        # precio_1 = 100 * (1 + r_1)
        # precio_2 = precio_1 * (1 + r_2) = 100 * (1 + r_1) * (1 + r_2)
        # Usando cumprod sobre (1 + retornos) y multiplicando por 100
        precios_cartera = (1 + retorno_cartera).cumprod() * 100.0
        
        # Crear DataFrame con el mismo índice que los precios
        df_precios = pd.DataFrame(
            {'Cartera': precios_cartera},
            index=self.precios.index
        )
        
        return df_precios
    
    def _inferir_cadencia(self) -> str:
        """
        Inferir la cadencia (frecuencia) de los datos basándose en el índice datetime
        
        Retorna:
        - 'diario', 'semanal', 'mensual', 'trimestral', o 'anual'
        """
        if len(self.precios) < 2:
            return 'diario'  # Por defecto
        
        # Calcular diferencias de días entre períodos consecutivos
        diferencias = self.precios.index.to_series().diff().dropna()
        dias_promedio = diferencias.dt.days.mean()
        
        # Determinar cadencia basándose en el promedio de días
        if dias_promedio <= 2:
            return 'diario'
        elif dias_promedio <= 8:
            return 'semanal'
        elif dias_promedio <= 40:
            return 'mensual'
        elif dias_promedio <= 100:
            return 'trimestral'
        else:
            return 'anual'
    
    def report(self) -> str:
        """
        Generar un reporte con métricas clave de la cartera en formato Markdown
        
        Calcula los precios de la cartera y retorna una cadena Markdown con:
        - Retorno promedio anualizado (usando retornos logarítmicos)
        - Retorno acumulado al final del período
        - Volatilidad anualizada
        - Número de activos
        - Número de períodos
        
        Retorna:
        - Cadena de texto en formato Markdown con las métricas de la cartera
        """
        # Calcular precios de la cartera
        precios_cartera = self.calcular_precios_cartera()
        
        # Inferir cadencia para los cálculos anualizados
        cadencia = self._inferir_cadencia()
        
        # Factores de anualización según la cadencia (consistentes con Operaciones.volatilidad_anualizada)
        factores_anualizacion = {
            'diario': 256,
            'semanal': 52,
            'mensual': 12,
            'trimestral': 4,
            'anual': 1
        }
        factor_anual = factores_anualizacion.get(cadencia, 256)
        
        # Calcular retornos logarítmicos de la cartera
        retornos_log = Operaciones.retornos_logaritmicos(precios_cartera)
        
        # Calcular retorno promedio anualizado (promedio de retornos log * factor anual)
        retorno_promedio_anualizado = retornos_log.mean().iloc[0] * factor_anual
        
        # Calcular retorno acumulado al final del período
        # Retorno acumulado = (precio_final / precio_inicial) - 1
        precio_inicial = precios_cartera.iloc[0, 0]
        precio_final = precios_cartera.iloc[-1, 0]
        retorno_acumulado = (precio_final / precio_inicial) - 1
        
        # Calcular volatilidad anualizada
        volatilidad_anual = Operaciones.volatilidad_anualizada(precios_cartera, cadencia=cadencia)
        volatilidad_anualizada = volatilidad_anual.iloc[0]
        
        # Obtener número de activos y períodos
        n_activos = len(self.pesos)
        n_periodos = len(self.precios)
        
        # Obtener fechas de inicio y fin del período
        fecha_inicio = self.precios.index[0].strftime('%Y-%m-%d')
        fecha_fin = self.precios.index[-1].strftime('%Y-%m-%d')
        
        # Crear reporte en formato Markdown
        markdown = f"""# Reporte de Cartera

## Período de Análisis
- **Fecha Inicio**: {fecha_inicio}
- **Fecha Fin**: {fecha_fin}
- **Cadencia**: {cadencia.capitalize()}

## Métricas de Rendimiento

| Métrica | Valor |
|---------|-------|
| **Retorno Promedio Anualizado (%)** | {retorno_promedio_anualizado * 100:.2f}% |
| **Retorno Acumulado (%)** | {retorno_acumulado * 100:.2f}% |
| **Volatilidad Anualizada (%)** | {volatilidad_anualizada * 100:.2f}% |

## Composición de la Cartera

| Detalle | Valor |
|---------|-------|
| **Número de Activos** | {n_activos} |
| **Número de Períodos** | {n_periodos} |
"""
        
        return markdown


class sim_mont:
    """
    Clase para realizar simulaciones de Monte Carlo
    
    Atributos:
    - n_simulaciones: Número de simulaciones a realizar
    - horizonte: Número de períodos a simular
    - retorno_esperado: Retorno esperado (drift) - puede ser un número o un array
    - volatilidad: Volatilidad anualizada - puede ser un número o un array
    - precio_inicial: Precio o valor inicial (por defecto 100)
    - cadencia: Frecuencia de los datos ('diario', 'semanal', 'mensual', etc.)
    """
    
    def __init__(self, n_simulaciones: int, horizonte: int, retorno_esperado: Union[float, np.ndarray, pd.Series], 
                 volatilidad: Union[float, np.ndarray, pd.Series], precio_inicial: float = 100.0, 
                 cadencia: str = 'diario', semilla: Optional[int] = None):
        """
        Inicializar una simulación de Monte Carlo
        
        Parámetros:
        - n_simulaciones: Número de simulaciones a realizar
        - horizonte: Número de períodos a simular
        - retorno_esperado: Retorno esperado (drift). Puede ser un escalar o un array/Series con un valor por período
        - volatilidad: Volatilidad anualizada. Puede ser un escalar o un array/Series con un valor por período
        - precio_inicial: Precio o valor inicial (por defecto 100.0)
        - cadencia: Frecuencia de los datos para calcular el factor de ajuste temporal ('diario', 'semanal', 'mensual', 'trimestral', 'anual')
        - semilla: Semilla opcional para reproducibilidad del generador aleatorio
        """
        # Validaciones
        if n_simulaciones <= 0:
            raise ValueError("n_simulaciones debe ser mayor que 0")
        if horizonte <= 0:
            raise ValueError("horizonte debe ser mayor que 0")
        if precio_inicial <= 0:
            raise ValueError("precio_inicial debe ser mayor que 0")
        
        self.n_simulaciones = n_simulaciones
        self.horizonte = horizonte
        self.precio_inicial = precio_inicial
        self.cadencia = cadencia.lower()
        
        # Factor de ajuste temporal para convertir volatilidad anualizada a volatilidad del período
        factores = {
            'diario': 252,
            'semanal': 52,
            'mensual': 12,
            'trimestral': 4,
            'anual': 1
        }
        self.periodos_por_anio = factores.get(self.cadencia, 252)
        
        # Convertir retorno_esperado y volatilidad a arrays si son escalares
        if isinstance(retorno_esperado, (int, float)):
            self.retorno_esperado = np.full(horizonte, retorno_esperado)
        elif isinstance(retorno_esperado, pd.Series):
            if len(retorno_esperado) != horizonte:
                raise ValueError(f"El tamaño de retorno_esperado ({len(retorno_esperado)}) debe coincidir con horizonte ({horizonte})")
            self.retorno_esperado = retorno_esperado.values
        elif isinstance(retorno_esperado, np.ndarray):
            if len(retorno_esperado) != horizonte:
                raise ValueError(f"El tamaño de retorno_esperado ({len(retorno_esperado)}) debe coincidir con horizonte ({horizonte})")
            self.retorno_esperado = retorno_esperado
        else:
            raise TypeError("retorno_esperado debe ser un número, numpy array o pandas Series")
        
        if isinstance(volatilidad, (int, float)):
            self.volatilidad = np.full(horizonte, volatilidad)
        elif isinstance(volatilidad, pd.Series):
            if len(volatilidad) != horizonte:
                raise ValueError(f"El tamaño de volatilidad ({len(volatilidad)}) debe coincidir con horizonte ({horizonte})")
            self.volatilidad = volatilidad.values
        elif isinstance(volatilidad, np.ndarray):
            if len(volatilidad) != horizonte:
                raise ValueError(f"El tamaño de volatilidad ({len(volatilidad)}) debe coincidir con horizonte ({horizonte})")
            self.volatilidad = volatilidad
        else:
            raise TypeError("volatilidad debe ser un número, numpy array o pandas Series")
        
        # Establecer semilla si se proporciona
        if semilla is not None:
            np.random.seed(semilla)
    
    def __repr__(self):
        """Representación de la simulación"""
        return (
            f"sim_mont(n_simulaciones={self.n_simulaciones}, "
            f"horizonte={self.horizonte}, "
            f"precio_inicial={self.precio_inicial}, "
            f"cadencia='{self.cadencia}')"
        )
    
    def simular(self) -> pd.DataFrame:
        """
        Ejecutar la simulación de Monte Carlo de precios
        
        Retorna:
        - DataFrame con n_simulaciones columnas y horizonte filas
          Cada columna representa una simulación diferente de precios
        """
        # Convertir volatilidad anualizada a volatilidad del período
        volatilidad_periodo = self.volatilidad / np.sqrt(self.periodos_por_anio)
        
        # Convertir retorno anualizado a retorno del período
        retorno_periodo = self.retorno_esperado / self.periodos_por_anio
        
        # Generar números aleatorios normales: shape (n_simulaciones, horizonte)
        # Cada fila es una simulación, cada columna es un período
        random_shocks = np.random.normal(0, 1, size=(self.n_simulaciones, self.horizonte))
        
        # Calcular retornos simples simulados
        # retorno_t = retorno_periodo_t + volatilidad_periodo_t * random_shock_t
        retornos_simulados = retorno_periodo + volatilidad_periodo * random_shocks
        
        # Calcular precios usando operaciones matriciales
        # precio_t = precio_inicial * (1 + r_1) * (1 + r_2) * ... * (1 + r_t)
        # Usar cumprod para encadenar los factores
        factores_crecimiento = 1 + retornos_simulados
        precios_simulados = self.precio_inicial * np.cumprod(factores_crecimiento, axis=1)
        
        # Añadir el precio inicial como primera fila para que todas empiecen en precio_inicial
        # Crear matriz completa: [precio_inicial, precios_simulados]
        precios_completos = np.column_stack([
            np.full((self.n_simulaciones, 1), self.precio_inicial),  # Primera columna: precio_inicial
            precios_simulados  # Resto de períodos
        ])
        
        # Crear DataFrame con índices para períodos y columnas para simulaciones
        # horizonte + 1 filas: período 0 (inicial) + períodos 1 a horizonte
        df_simulacion = pd.DataFrame(
            precios_completos.T,  # Transponer para que filas = períodos, columnas = simulaciones
            columns=[f'Sim_{i+1}' for i in range(self.n_simulaciones)],
            index=pd.RangeIndex(start=0, stop=self.horizonte + 1, name='Periodo')
        )
        
        return df_simulacion
    
    def visualizar(self, df_simulacion: Optional[pd.DataFrame] = None, mostrar_todas: bool = True, 
                   mostrar_percentiles: bool = True, mostrar_media: bool = True, 
                   percentiles: List[float] = [5, 25, 50, 75, 95], 
                   figsize: tuple = (12, 6), titulo: Optional[str] = None) -> None:
        """
        Visualizar los resultados de la simulación de Monte Carlo usando seaborn
        
        Parámetros:
        - df_simulacion: DataFrame con los resultados de la simulación (si None, se ejecuta simular())
        - mostrar_todas: Si True, muestra todas las trayectorias individuales (puede ser lento con muchas simulaciones)
        - mostrar_percentiles: Si True, muestra líneas de percentiles
        - mostrar_media: Si True, muestra la media de todas las simulaciones
        - percentiles: Lista de percentiles a mostrar (por defecto [5, 25, 50, 75, 95])
        - figsize: Tamaño de la figura (ancho, alto)
        - titulo: Título personalizado para el gráfico
        """
        # Si no se proporciona el DataFrame, ejecutar la simulación
        if df_simulacion is None:
            df_simulacion = self.simular()
        
        # Configurar el estilo de seaborn
        sns.set_style("whitegrid")
        plt.figure(figsize=figsize)
        
        # Preparar datos para visualización
        df_plot = df_simulacion.reset_index()
        
        # Mostrar todas las trayectorias si se solicita y hay pocas simulaciones
        if mostrar_todas and self.n_simulaciones <= 100:
            for col in df_simulacion.columns:
                sns.lineplot(data=df_plot, x='Periodo', y=col, alpha=0.3, linewidth=0.5, color='lightblue')
        elif mostrar_todas and self.n_simulaciones > 100:
            # Si hay muchas simulaciones, mostrar solo una muestra
            muestra = np.random.choice(df_simulacion.columns, size=100, replace=False)
            for col in muestra:
                sns.lineplot(data=df_plot, x='Periodo', y=col, alpha=0.2, linewidth=0.3, color='lightblue')
        
        # Calcular y mostrar percentiles
        if mostrar_percentiles:
            percentiles_valores = [p/100 for p in percentiles]
            percentiles_df = df_simulacion.quantile(percentiles_valores, axis=1).T.reset_index()
            for pct in percentiles:
                pct_valor = pct / 100
                label = f'{pct}%' if pct != 50 else 'Mediana (50%)'
                color = 'red' if pct == 50 else ('darkblue' if pct in [25, 75] else 'blue')
                linewidth = 2.0 if pct == 50 else 1.5
                sns.lineplot(data=percentiles_df, x='Periodo', y=pct_valor, 
                           label=label, linewidth=linewidth, color=color, linestyle='--' if pct != 50 else '-')
        
        # Calcular y mostrar la media
        if mostrar_media:
            media = df_simulacion.mean(axis=1)
            media_df = pd.DataFrame({'Periodo': df_plot['Periodo'], 'Media': media.values})
            sns.lineplot(data=media_df, x='Periodo', y='Media', 
                       label='Media', linewidth=2.5, color='green', linestyle='-')
        
        # Configurar etiquetas y título
        if titulo is None:
            titulo = f'Simulación de Monte Carlo\n({self.n_simulaciones} simulaciones, horizonte: {self.horizonte}, precio inicial: {self.precio_inicial})'
        
        plt.title(titulo, fontsize=14, fontweight='bold')
        plt.xlabel('Período', fontsize=12)
        plt.ylabel('Precio', fontsize=12)
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # Ajustar layout y mostrar
        plt.tight_layout()
        plt.show()


class Graficos:
    """
    Clase para generar gráficos financieros usando seaborn y matplotlib
    
    Métodos:
    - grafico_lineas: Muestra la trayectoria de los activos
    - grafico_volatilidad_rentabilidad: Scatterplot de volatilidad vs rentabilidad
    - matriz_correlaciones: Matriz de correlaciones con heatmap
    """
    
    @staticmethod
    def grafico_lineas(precios: pd.DataFrame, figsize: tuple = (12, 6), 
                       titulo: Optional[str] = None, mostrar_leyenda: bool = True) -> None:
        """
        Gráfico de líneas que muestra la trayectoria de los activos
        
        Parámetros:
        - precios: DataFrame con precios históricos (índice datetime, columnas = activos)
        - figsize: Tamaño de la figura (ancho, alto)
        - titulo: Título personalizado para el gráfico
        - mostrar_leyenda: Si True, muestra la leyenda
        """
        if precios.empty:
            raise ValueError("El DataFrame de precios está vacío")
        
        # Configurar estilo
        sns.set_style("whitegrid")
        plt.figure(figsize=figsize)
        
        # Preparar datos: resetear índice para que la fecha sea una columna
        df_plot = precios.reset_index()
        fecha_col = df_plot.columns[0]  # Primera columna es el índice (fecha)
        
        # Graficar cada activo
        columnas_activos = precios.columns.tolist()
        for col in columnas_activos:
            sns.lineplot(data=df_plot, x=fecha_col, y=col, label=col)
        
        # Configurar título
        if titulo is None:
            titulo = 'Trayectoria de Precios de los Activos'
        
        plt.title(titulo, fontsize=14, fontweight='bold')
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Precio', fontsize=12)
        
        if mostrar_leyenda:
            plt.legend(title='Activos', loc='best', fontsize=9)
        
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def grafico_volatilidad_rentabilidad(df: pd.DataFrame, col_volatilidad: str, 
                                        col_rentabilidad: str, figsize: tuple = (10, 6),
                                        titulo: Optional[str] = None,
                                        etiquetar_puntos: bool = False) -> None:
        """
        Scatterplot con volatilidad (X) vs rentabilidad (Y)
        
        Parámetros:
        - df: DataFrame con al menos dos columnas: una de volatilidad y otra de rentabilidad
        - col_volatilidad: Nombre de la columna que contiene la volatilidad
        - col_rentabilidad: Nombre de la columna que contiene la rentabilidad
        - figsize: Tamaño de la figura (ancho, alto)
        - titulo: Título personalizado para el gráfico
        - etiquetar_puntos: Si True, etiqueta cada punto con el nombre del índice
        """
        if col_volatilidad not in df.columns:
            raise ValueError(f"La columna '{col_volatilidad}' no existe en el DataFrame")
        if col_rentabilidad not in df.columns:
            raise ValueError(f"La columna '{col_rentabilidad}' no existe en el DataFrame")
        
        # Configurar estilo
        sns.set_style("whitegrid")
        plt.figure(figsize=figsize)
        
        # Crear scatterplot
        sns.scatterplot(data=df, x=col_volatilidad, y=col_rentabilidad, s=100, alpha=0.7)
        
        # Etiquetar puntos si se solicita
        if etiquetar_puntos:
            for idx, row in df.iterrows():
                plt.annotate(str(idx), (row[col_volatilidad], row[col_rentabilidad]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Configurar título y etiquetas
        if titulo is None:
            titulo = 'Rentabilidad vs Volatilidad'
        
        plt.title(titulo, fontsize=14, fontweight='bold')
        plt.xlabel('Volatilidad (%)', fontsize=12)
        plt.ylabel('Rentabilidad (%)', fontsize=12)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def matriz_correlaciones(precios: pd.DataFrame, figsize: tuple = (10, 8),
                            titulo: Optional[str] = None, 
                            annot: bool = False, fmt: str = '.2f',
                            cmap: str = 'coolwarm', vmin: float = -1, vmax: float = 1) -> pd.DataFrame:
        """
        Matriz de correlaciones con heatmap visual
        
        Parámetros:
        - precios: DataFrame con precios históricos (índice datetime, columnas = activos)
        - figsize: Tamaño de la figura (ancho, alto)
        - titulo: Título personalizado para el gráfico
        - annot: Si True, muestra los valores de correlación en cada celda
        - fmt: Formato para los valores anotados
        - cmap: Mapa de colores para el heatmap
        - vmin, vmax: Límites del mapa de colores
        
        Retorna:
        - DataFrame con la matriz de correlaciones
        """
        if precios.empty:
            raise ValueError("El DataFrame de precios está vacío")
        
        # Calcular retornos logarítmicos para obtener correlaciones de retornos
        retornos_log = Operaciones.retornos_logaritmicos(precios)
        
        # Calcular matriz de correlaciones
        matriz_corr = retornos_log.corr()
        
        # Configurar estilo
        sns.set_style("whitegrid")
        plt.figure(figsize=figsize)
        
        # Crear heatmap
        sns.heatmap(matriz_corr, annot=annot, fmt=fmt, cmap=cmap, 
                   vmin=vmin, vmax=vmax, center=0, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        # Configurar título
        if titulo is None:
            titulo = 'Matriz de Correlaciones'
        
        plt.title(titulo, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        return matriz_corr
