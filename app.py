import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import base64

# Configuración de la página
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# Funciones de utilidad
def calculate_metrics(returns, weights, rf_rate):
    """
    Calcula las métricas principales del portafolio:
    - Retorno esperado anualizado
    - Volatilidad anualizada
    - Ratio de Sharpe
    - Value at Risk (VaR) al 95%
    - Conditional Value at Risk (CVaR) al 95%
    - Máximo Drawdown histórico
    """
    port_ret = np.sum(returns.mean() * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe = (port_ret - rf_rate) / port_vol if port_vol != 0 else 0
    
    # Calcular VaR y CVaR
    port_returns = returns.dot(weights)
    
    # Verificar que hay datos suficientes
    if len(port_returns) == 0:
        return port_ret, port_vol, sharpe, 0, 0, 0
    
    var_95 = np.percentile(port_returns, 5)
    cvar_95 = port_returns[port_returns <= var_95].mean()
    
    # Maximum Drawdown
    cum_returns = (1 + port_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns/rolling_max - 1
    max_drawdown = drawdowns.min()
    
    return port_ret, port_vol, sharpe, var_95, cvar_95, max_drawdown

def monte_carlo_var_cvar(returns, weights, num_simulations=10000, confidence_level=0.95):
    """
    Calcula VaR y CVaR usando simulación Monte Carlo
    Retorna: var, cvar, y el array de simulaciones
    """
    # Calcular media y covarianza de retornos
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Generar simulaciones
    portfolio_sims = np.random.multivariate_normal(
        mean_returns, 
        cov_matrix, 
        num_simulations
    )
    
    # Calcular retornos del portfolio para cada simulación
    portfolio_returns = portfolio_sims.dot(weights)
    
    # Calcular VaR y CVaR
    var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    
    return var, cvar, portfolio_returns

def hierarchical_risk_parity(returns):
    """
    Implementa el algoritmo Hierarchical Risk Parity (HRP)
    """
    # Calcular matriz de correlación
    corr = returns.corr()
    
    # Convertir correlación a distancia
    dist = np.sqrt((1 - corr) / 2)
    
    # Clustering jerárquico
    link = linkage(squareform(dist.values), method='single')
    
    # Obtener orden de clustering
    sort_idx = dendrogram(link, no_plot=True)['leaves']
    
    # Calcular pesos usando risk parity en clusters
    weights = np.ones(len(returns.columns))
    
    def _get_cluster_var(cov, cluster_items):
        cov_slice = cov.iloc[cluster_items, cluster_items]
        ivp = 1 / np.diag(cov_slice)
        ivp /= ivp.sum()
        w = ivp.reshape(-1, 1)
        cluster_var = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
        return cluster_var
    
    def _recursive_bisection(cov, sort_idx):
        weights = pd.Series(1, index=sort_idx)
        cluster_items = [sort_idx]
        
        while len(cluster_items) > 0:
            cluster_items = [i[j:k] for i in cluster_items 
                           for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) 
                           if len(i) > 1]
            
            for i in range(0, len(cluster_items), 2):
                cluster0 = cluster_items[i]
                cluster1 = cluster_items[i + 1]
                
                var0 = _get_cluster_var(cov, cluster0)
                var1 = _get_cluster_var(cov, cluster1)
                
                alpha = 1 - var0 / (var0 + var1)
                
                weights[cluster0] *= alpha
                weights[cluster1] *= 1 - alpha
                
        return weights
    
    cov = returns.cov()
    weights = _recursive_bisection(cov, sort_idx)
    
    return weights.values / weights.sum()

def get_stock_data(tickers, start_date, end_date):
    """
    Descarga datos históricos de precios para los tickers seleccionados
    utilizando la API de Yahoo Finance.
    """
    data = pd.DataFrame()
    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, multi_level_index=False, auto_adjust=False)['Adj Close']
            data[ticker] = stock_data
        except:
            st.error(f"❌ Error downloading data for {ticker}")
    return data

def calculate_rolling_beta(returns, market_returns, window=60):
    """
    Calcula el beta rolling contra el mercado
    """
    rolling_cov = returns.rolling(window=window).cov(market_returns)
    rolling_var = market_returns.rolling(window=window).var()
    rolling_beta = rolling_cov / rolling_var
    return rolling_beta

def calculate_rolling_correlation(returns1, returns2, window=60):
    """
    Calcula la correlación rolling entre dos series de retornos
    """
    return returns1.rolling(window=window).corr(returns2)

# Función para crear el footer
def add_footer():
    footer_html = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #FAFAFA;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #333;
    }
    </style>
    <div class="footer">
        💼 Made with ❤️ by Fede Martinez - Finanzas & Data | Enhanced with Portfolio Management Concepts
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# Título y descripción con emojis
st.title("📊 Advanced Portfolio Optimizer")
st.markdown("""
Esta aplicación te permite simular y optimizar portafolios de inversión utilizando la teoría moderna de portafolios.
Incluye costos de transacción, slippage, métricas avanzadas de riesgo y análisis de correlación para un análisis completo.

### ¿Cómo funciona? 🧠
1. Ingresa los símbolos de tus acciones preferidas
2. Ajusta los parámetros de inversión
3. Haz clic en "Run Optimization" para generar portafolios óptimos con dos metodologías

### Conceptos clave 🔑
- **Ratio de Sharpe**: Mide el rendimiento ajustado por riesgo (mayor es mejor)
- **VaR Monte Carlo**: Pérdida máxima esperada con 95% de confianza usando simulaciones
- **CVaR Monte Carlo**: Pérdida promedio esperada en los peores escenarios
- **HRP**: Hierarchical Risk Parity - método avanzado de diversificación basado en clustering
- **Retorno Real**: Rendimiento ajustado por inflación
""")

# Sidebar - Parámetros de entrada
with st.sidebar:
    st.header("🛠️ Portfolio Parameters")
    
    # Input para tickers con explicación
    st.markdown("**Símbolos de acciones** 📈")
    default_tickers = "SMH,GLD,NU,AMZN,VIST,PAM,JNJ,GGAL,BABA"
    tickers_input = st.text_input("Ingresa símbolos bursátiles (separados por comas, máx. 15):", 
                                 value=default_tickers,
                                 help="Ejemplos: AAPL (Apple), MSFT (Microsoft), GOOGL (Google)")
    stocks = [x.strip() for x in tickers_input.split(',')]
    
    if len(stocks) > 15:
        st.error("⚠️ Máximo 15 acciones permitidas")
        stocks = stocks[:15]
    
    # Fechas con mejor explicación
    st.markdown("**Periodo de análisis** 📅")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha inicial", 
                                  datetime.now() - timedelta(days=365*2),
                                  help="Recomendado: mínimo 2 años de datos históricos")
    with col2:
        end_date = st.date_input("Fecha final", 
                                datetime.now())
    
    # Parámetros financieros con explicaciones
    st.markdown("**Parámetros financieros** 💰")
    initial_capital = st.number_input("Capital inicial ($)", 
                                    min_value=1000, 
                                    value=10000,
                                    help="Monto total a invertir")
    
    rf_rate = st.slider("Tasa libre de riesgo (%) 🏦", 
                       min_value=0.0, 
                       max_value=10.0, 
                       value=4.0,
                       help="Rendimiento de bonos del tesoro o similar") / 100
    
    inflation_rate = st.slider("Tasa de inflación esperada (%) 📊", 
                              min_value=0.0, 
                              max_value=15.0, 
                              value=3.0,
                              help="Inflación anual esperada para calcular retornos reales") / 100
    
    transaction_cost = st.slider("Costo de transacción (%) 💸", 
                               min_value=0.0, 
                               max_value=2.0, 
                               value=0.1,
                               help="Comisiones cobradas por el broker") / 100
    
    slippage = st.slider("Slippage (%) 📉", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.1,
                        help="Diferencia entre precio esperado y ejecutado") / 100
    
    # Parámetros de simulación
    st.markdown("**Parámetros de simulación** 🔢")
    num_simulations = st.slider("Número de simulaciones (Markowitz)", 
                              min_value=100, 
                              max_value=5000, 
                              value=1000,
                              help="Más simulaciones = mayor precisión pero más tiempo de ejecución")
    
    num_mc_sims = st.slider("Simulaciones Monte Carlo (VaR/CVaR)", 
                           min_value=1000, 
                           max_value=50000, 
                           value=10000,
                           help="Simulaciones para calcular VaR y CVaR")

# Main content
if st.button("🚀 Run Optimization"):
    with st.spinner("⏳ Descargando datos y ejecutando simulaciones..."):
        # Obtener datos
        stock_data = get_stock_data(stocks, start_date, end_date)
        
        if stock_data.empty:
            st.error("❌ No hay datos disponibles para las acciones y fechas seleccionadas")
        else:
            # Calcular retornos
            returns = stock_data.pct_change().dropna()
            
            # Mostrar información sobre los datos
            st.info(f"ℹ️ Analizando {len(returns)} días de datos históricos para {len(stocks)} acciones")
            
            # Arrays para almacenar resultados
            all_weights = np.zeros((num_simulations, len(stocks)))
            metrics = np.zeros((num_simulations, 6))  # Ret, Vol, Sharpe, VaR, CVaR, MaxDD
            
            # Explicación de la simulación
            st.markdown("### 🔄 Simulación Monte Carlo (Optimización Markowitz)")
            st.markdown("""
            Estamos generando miles de portafolios con diferentes combinaciones de pesos
            para encontrar la distribución óptima que maximiza el ratio de Sharpe.
            """)
            
            progress_bar = st.progress(0)
            
            # Simulación Monte Carlo
            for port in range(num_simulations):
                weights = np.random.random(len(stocks))
                weights = weights/np.sum(weights)
                all_weights[port,:] = weights
                
                metrics[port,:] = calculate_metrics(returns, weights, rf_rate)
                
                # Actualizar barra de progreso cada 5%
                if port % (num_simulations//20) == 0:
                    progress_bar.progress(port/num_simulations)
            
            progress_bar.progress(1.0)
            
            # Crear DataFrame con resultados
            results = pd.DataFrame(metrics, 
                                 columns=['Return', 'Volatility', 'Sharpe', 
                                        'VaR_95', 'CVaR_95', 'Max_Drawdown'])
            
            # Encontrar portafolio óptimo (Markowitz)
            optimal_idx = results['Sharpe'].argmax()
            optimal_weights = all_weights[optimal_idx,:]
            
            # Calcular portafolio HRP
            st.markdown("### 🌳 Calculando Hierarchical Risk Parity (HRP)")
            hrp_weights = hierarchical_risk_parity(returns)
            
            # Calcular métricas para HRP
            hrp_metrics = calculate_metrics(returns, hrp_weights, rf_rate)
            
            st.success("✅ Optimización completada con éxito!")
            
            # ==================== SECCIÓN 1: MARKOWITZ ====================
            st.markdown("---")
            st.markdown("## 📈 SECCIÓN 1: Portafolio Óptimo (Markowitz - Máximo Sharpe)")
            
            # Display results in multiple columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📈 Métricas del Portafolio")
                
                # Calcular retorno real
                real_return = results.iloc[optimal_idx]['Return'] - inflation_rate
                
                metrics_df = pd.DataFrame({
                    'Métrica': ['Retorno Esperado (Nominal)', 'Retorno Real (Ajust. Inflación)',
                               'Volatilidad', 'Ratio de Sharpe',
                             'VaR (95%)', 'CVaR (95%)', 'Máximo Drawdown'],
                    'Valor': [f"{results.iloc[optimal_idx]['Return']*100:.2f}%",
                             f"{real_return*100:.2f}%",
                             f"{results.iloc[optimal_idx]['Volatility']*100:.2f}%",
                             f"{results.iloc[optimal_idx]['Sharpe']:.2f}",
                             f"{results.iloc[optimal_idx]['VaR_95']*100:.2f}%",
                             f"{results.iloc[optimal_idx]['CVaR_95']*100:.2f}%",
                             f"{results.iloc[optimal_idx]['Max_Drawdown']*100:.2f}%"]
                })
                st.dataframe(metrics_df)
                
                # Explicación de métricas
                with st.expander("ℹ️ Explicación de métricas"):
                    st.markdown(f"""
                    - **Retorno Nominal**: Rendimiento anualizado proyectado sin ajustes
                    - **Retorno Real**: Retorno ajustado por inflación ({inflation_rate*100:.1f}%)
                    - **Volatilidad**: Medida de riesgo (desviación estándar anualizada)
                    - **Ratio de Sharpe**: Retorno ajustado por riesgo (mayor = mejor)
                    - **VaR (95%)**: Pérdida máxima diaria con 95% de confianza
                    - **CVaR (95%)**: Pérdida promedio en el peor 5% de los escenarios
                    - **Máximo Drawdown**: Caída máxima desde un pico
                    """)
            
            with col2:
                st.subheader("💼 Pesos Óptimos (Markowitz)")
                weights_df = pd.DataFrame({
                    'Acción': stocks,
                    'Peso': [f"{w*100:.2f}%" for w in optimal_weights],
                    'Valor ($)': [f"${w*initial_capital:,.2f}" for w in optimal_weights]
                })
                st.dataframe(weights_df)
                
                # Gráfico de pesos
                fig_weights = px.pie(
                    names=stocks,
                    values=optimal_weights,
                    title="Distribución del Portafolio (Markowitz)"
                )
                st.plotly_chart(fig_weights, use_container_width=True)
            
            with col3:
                st.subheader("💸 Costos de Transacción")
                transaction_cost_value = initial_capital * transaction_cost
                slippage_value = initial_capital * slippage
                total_cost = transaction_cost_value + slippage_value
                
                st.metric("Comisiones", f"${transaction_cost_value:,.2f}")
                st.metric("Slippage estimado", f"${slippage_value:,.2f}")
                st.metric("Costo total", f"${total_cost:,.2f}")
                st.metric("Inversión efectiva", f"${initial_capital-total_cost:,.2f}")
                
                # Explicación
                with st.expander("ℹ️ Sobre los costos"):
                    st.markdown("""
                    Los costos totales reducen tu capital invertido efectivo.
                    - **Comisiones**: Pagos al broker por ejecutar órdenes
                    - **Slippage**: Diferencia entre precio esperado y ejecutado
                    """)
            
            # VaR/CVaR Monte Carlo para Markowitz
            st.markdown("### 🎲 Análisis VaR/CVaR con Monte Carlo (Markowitz)")
            
            with st.spinner("Ejecutando simulaciones Monte Carlo..."):
                mc_var, mc_cvar, mc_returns = monte_carlo_var_cvar(
                    returns, optimal_weights, num_mc_sims
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("VaR 95% (Monte Carlo)", f"{mc_var*100:.2f}%")
                st.metric("CVaR 95% (Monte Carlo)", f"{mc_cvar*100:.2f}%")
                
                with st.expander("ℹ️ Sobre VaR/CVaR Monte Carlo"):
                    st.markdown(f"""
                    Estos valores se calcularon mediante **{num_mc_sims:,} simulaciones** Monte Carlo,
                    generando escenarios aleatorios basados en la distribución histórica de retornos.
                    
                    - **VaR**: En el peor 5% de los casos, puedes perder al menos {mc_var*100:.2f}% en un día
                    - **CVaR**: La pérdida promedio en ese peor 5% de casos es {mc_cvar*100:.2f}%
                    """)
            
            with col2:
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Histogram(
                    x=mc_returns*100,
                    nbinsx=50,
                    name='Distribución de Retornos',
                    marker_color='lightblue'
                ))
                
                fig_mc.add_vline(x=mc_var*100, line_dash="dash", line_color="red",
                               annotation_text=f"VaR 95%: {mc_var*100:.2f}%")
                fig_mc.add_vline(x=mc_cvar*100, line_dash="dash", line_color="darkred",
                               annotation_text=f"CVaR 95%: {mc_cvar*100:.2f}%")
                
                fig_mc.update_layout(
                    title="Distribución de Retornos (Monte Carlo)",
                    xaxis_title="Retorno (%)",
                    yaxis_title="Frecuencia",
                    height=400
                )
                
                st.plotly_chart(fig_mc, use_container_width=True)
            
            # Heatmap de correlación
            st.markdown("### 🔥 Matriz de Correlación de Activos")
            
            with st.expander("ℹ️ Interpretando la matriz de correlación"):
                st.markdown("""
                - **1.0 (rojo)**: Correlación perfecta positiva - los activos se mueven juntos
                - **0.0 (amarillo)**: Sin correlación - movimientos independientes
                - **-1.0 (azul)**: Correlación perfecta negativa - se mueven en direcciones opuestas
                
                La diversificación es más efectiva cuando los activos tienen baja correlación entre sí.
                """)
            
            corr_matrix = returns.corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                colorbar=dict(title="Correlación")
            ))
            
            fig_corr.update_layout(
                title="Matriz de Correlación entre Activos",
                height=600,
                xaxis_title="Activo",
                yaxis_title="Activo"
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Visualizaciones
            st.markdown("### 📊 Visualizaciones Adicionales")
            
            # Efficient Frontier Plot
            st.subheader("🎯 Frontera Eficiente")
            
            with st.expander("ℹ️ ¿Qué es la Frontera Eficiente?"):
                st.markdown("""
                La **Frontera Eficiente** muestra todos los portafolios posibles en el espacio riesgo-retorno.
                - **Eje X**: Volatilidad (riesgo)
                - **Eje Y**: Retorno esperado
                - **Color**: Ratio de Sharpe (más brillante = mejor)
                - **Estrella roja**: Portafolio óptimo que maximiza el ratio de Sharpe
                
                Los mejores portafolios están en la parte superior izquierda (mayor retorno con menor riesgo).
                """)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['Volatility'],
                y=results['Return'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=results['Sharpe'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Ratio de Sharpe")
                ),
                text=[f"Retorno: {r:.2%}<br>Volatilidad: {v:.2%}<br>Sharpe: {s:.2f}" 
                      for r,v,s in zip(results['Return'], 
                                     results['Volatility'], 
                                     results['Sharpe'])],
                name="Portafolios"
            ))
            
            # Añadir punto óptimo
            fig.add_trace(go.Scatter(
                x=[results.iloc[optimal_idx]['Volatility']],
                y=[results.iloc[optimal_idx]['Return']],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name="Portafolio Óptimo (Markowitz)"
            ))
            
            fig.update_layout(
                title="Frontera Eficiente de Markowitz",
                xaxis_title="Volatilidad (Riesgo)",
                yaxis_title="Retorno Esperado",
                showlegend=True,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Historical Performance
            st.subheader("📈 Análisis de Rendimiento Histórico")
            
            with st.expander("ℹ️ Sobre el rendimiento histórico"):
                st.markdown("""
                Este gráfico muestra cómo se habría comportado el portafolio óptimo durante el período histórico analizado.
                - La línea muestra el crecimiento de $1 invertido al inicio del período
                - Pendiente positiva = ganancias
                - Pendiente negativa = pérdidas
                """)
            
            portfolio_returns = returns.dot(optimal_weights)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode='lines',
                name='Valor del Portafolio',
                line=dict(color='rgb(0, 128, 255)', width=2)
            ))
            
            # Agregar línea de referencia
            fig2.add_hline(y=1, line_dash="dash", line_color="gray", 
                         annotation_text="Inversión inicial")
            
            fig2.update_layout(
                title="Rendimiento Histórico del Portafolio (Markowitz)",
                xaxis_title="Fecha",
                yaxis_title="Retorno Acumulado",
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Drawdown analysis
            st.subheader("📉 Análisis de Drawdown")
            
            with st.expander("ℹ️ ¿Qué es un Drawdown?"):
                st.markdown("""
                El **Drawdown** muestra la caída desde el máximo anterior.
                - 0% significa que estamos en máximos históricos
                - -10% significa que hemos caído un 10% desde el máximo anterior
                - Es un indicador importante del riesgo real que enfrentarías como inversor
                """)
            
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns/rolling_max - 1
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=drawdowns.index,
                y=drawdowns.values,
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='rgba(255, 0, 0, 0.5)')
            ))
            
            fig3.update_layout(
                title="Análisis de Drawdown del Portafolio (Markowitz)",
                xaxis_title="Fecha",
                yaxis_title="Drawdown",
                height=400
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Beta y Correlación Rolling (asumiendo SPY como mercado)
            st.markdown("### 📊 Análisis de Beta y Correlación Rolling")
            
            with st.expander("ℹ️ Sobre Beta y Correlación Rolling"):
                st.markdown("""
                - **Beta Rolling**: Mide la sensibilidad del portafolio respecto al mercado (SPY)
                  - Beta > 1: El portafolio es más volátil que el mercado
                  - Beta < 1: El portafolio es menos volátil que el mercado
                  
                - **Correlación Rolling**: Mide qué tan relacionados están los movimientos del portafolio con el mercado
                  - Correlación alta: Se mueven muy parecido
                  - Correlación baja: Movimientos más independientes
                """)
            
            # Intentar obtener datos de SPY como proxy del mercado
            try:
                spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False, multi_level_index=False, auto_adjust=False)['Adj Close']
                spy_returns = spy_data.pct_change().dropna()
                
                # Alinear fechas
                common_dates = portfolio_returns.index.intersection(spy_returns.index)
                portfolio_returns_aligned = portfolio_returns.loc[common_dates]
                spy_returns_aligned = spy_returns.loc[common_dates]
                
                # Calcular rolling beta y correlación
                rolling_beta = calculate_rolling_beta(portfolio_returns_aligned, spy_returns_aligned, window=60)
                rolling_corr = calculate_rolling_correlation(portfolio_returns_aligned, spy_returns_aligned, window=60)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_beta = go.Figure()
                    fig_beta.add_trace(go.Scatter(
                        x=rolling_beta.index,
                        y=rolling_beta.values,
                        mode='lines',
                        name='Beta Rolling (60 días)',
                        line=dict(color='purple')
                    ))
                    fig_beta.add_hline(y=1, line_dash="dash", line_color="gray",
                                     annotation_text="Beta = 1 (Mercado)")
                    fig_beta.update_layout(
                        title="Beta Rolling vs SPY (Mercado)",
                        xaxis_title="Fecha",
                        yaxis_title="Beta",
                        height=400
                    )
                    st.plotly_chart(fig_beta, use_container_width=True)
                
                with col2:
                    fig_corr_rolling = go.Figure()
                    fig_corr_rolling.add_trace(go.Scatter(
                        x=rolling_corr.index,
                        y=rolling_corr.values,
                        mode='lines',
                        name='Correlación Rolling (60 días)',
                        line=dict(color='orange')
                    ))
                    fig_corr_rolling.update_layout(
                        title="Correlación Rolling vs SPY",
                        xaxis_title="Fecha",
                        yaxis_title="Correlación",
                        height=400
                    )
                    st.plotly_chart(fig_corr_rolling, use_container_width=True)
                    
            except Exception as e:
                st.warning(f"⚠️ No se pudo calcular Beta y Correlación Rolling: {str(e)}")
            
            # Risk Metrics Distribution
            st.subheader("📊 Distribución de Métricas de Riesgo")
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("ℹ️ Sobre la distribución de retornos"):
                    st.markdown("""
                    Este histograma muestra la distribución de los retornos diarios:
                    - Forma de campana = distribución normal
                    - Cola izquierda larga = riesgo de caídas extremas
                    - Mayor concentración alrededor de 0 = menor volatilidad
                    """)
                
                fig4 = px.histogram(
                    portfolio_returns, 
                    title="Distribución de Retornos Diarios",
                    labels={'value': 'Retorno', 'count': 'Frecuencia'},
                    color_discrete_sequence=['rgb(0, 128, 255)']
                )
                
                # Agregar línea para VaR
                fig4.add_vline(x=results.iloc[optimal_idx]['VaR_95'], line_dash="dash", 
                             line_color="red", 
                             annotation_text="VaR 95%")
                
                st.plotly_chart(fig4, use_container_width=True)
            
            with col2:
                with st.expander("ℹ️ Sobre la volatilidad móvil"):
                    st.markdown("""
                    La **Volatilidad Móvil** muestra cómo ha variado el riesgo del portafolio:
                    - Picos = períodos de alta incertidumbre
                    - Volatilidad baja y estable = mercados calmos
                    - Se calcula como la desviación estándar móvil anualizada
                    """)
                
                rolling_vol = portfolio_returns.rolling(window=21).std() * np.sqrt(252)
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol.values,
                    mode='lines',
                    name='Volatilidad Móvil',
                    line=dict(color='rgb(75, 0, 130)')
                ))
                
                # Agregar línea de referencia
                fig5.add_hline(y=results.iloc[optimal_idx]['Volatility'], line_dash="dash", 
                             line_color="gray", 
                             annotation_text="Volatilidad Media")
                
                fig5.update_layout(title="Volatilidad Móvil de 21 Días (Anualizada)")
                st.plotly_chart(fig5, use_container_width=True)
            
            # ==================== SECCIÓN 2: HRP ====================
            st.markdown("---")
            st.markdown("## 🌳 SECCIÓN 2: Portafolio Hierarchical Risk Parity (HRP)")
            st.markdown("""
            El método **HRP** utiliza clustering jerárquico para agrupar activos similares y luego 
            asigna pesos usando un enfoque de paridad de riesgo. Esta técnica es más robusta ante 
            cambios en las correlaciones y puede ofrecer mejor diversificación.
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📈 Métricas HRP")
                
                # Calcular retorno real HRP
                real_return_hrp = hrp_metrics[0] - inflation_rate
                
                metrics_hrp_df = pd.DataFrame({
                    'Métrica': ['Retorno Esperado (Nominal)', 'Retorno Real (Ajust. Inflación)',
                               'Volatilidad', 'Ratio de Sharpe',
                             'VaR (95%)', 'CVaR (95%)', 'Máximo Drawdown'],
                    'Valor': [f"{hrp_metrics[0]*100:.2f}%",
                             f"{real_return_hrp*100:.2f}%",
                             f"{hrp_metrics[1]*100:.2f}%",
                             f"{hrp_metrics[2]:.2f}",
                             f"{hrp_metrics[3]*100:.2f}%",
                             f"{hrp_metrics[4]*100:.2f}%",
                             f"{hrp_metrics[5]*100:.2f}%"]
                })
                st.dataframe(metrics_hrp_df)
            
            with col2:
                st.subheader("💼 Pesos HRP")
                weights_hrp_df = pd.DataFrame({
                    'Acción': stocks,
                    'Peso': [f"{w*100:.2f}%" for w in hrp_weights],
                    'Valor ($)': [f"${w*initial_capital:,.2f}" for w in hrp_weights]
                })
                st.dataframe(weights_hrp_df)
                
                # Gráfico de pesos HRP
                fig_weights_hrp = px.pie(
                    names=stocks,
                    values=hrp_weights,
                    title="Distribución del Portafolio (HRP)"
                )
                st.plotly_chart(fig_weights_hrp, use_container_width=True)
            
            with col3:
                st.subheader("📊 Comparación de Métodos")
                comparison_df = pd.DataFrame({
                    'Métrica': ['Retorno Nominal', 'Retorno Real', 'Volatilidad', 'Sharpe Ratio'],
                    'Markowitz': [
                        f"{results.iloc[optimal_idx]['Return']*100:.2f}%",
                        f"{(results.iloc[optimal_idx]['Return'] - inflation_rate)*100:.2f}%",
                        f"{results.iloc[optimal_idx]['Volatility']*100:.2f}%",
                        f"{results.iloc[optimal_idx]['Sharpe']:.2f}"
                    ],
                    'HRP': [
                        f"{hrp_metrics[0]*100:.2f}%",
                        f"{real_return_hrp*100:.2f}%",
                        f"{hrp_metrics[1]*100:.2f}%",
                        f"{hrp_metrics[2]:.2f}"
                    ]
                })
                st.dataframe(comparison_df)
                
                with st.expander("ℹ️ ¿Cuál elegir?"):
                    st.markdown("""
                    - **Markowitz (Max Sharpe)**: Maximiza el retorno ajustado por riesgo, 
                      pero puede ser sensible a errores en las estimaciones
                    - **HRP**: Más robusto y estable, mejor diversificación estructural,
                      pero puede tener menor Sharpe Ratio
                    
                    Considera usar ambos enfoques para tomar una decisión más informada.
                    """)
            
            # VaR/CVaR Monte Carlo para HRP
            st.markdown("### 🎲 Análisis VaR/CVaR con Monte Carlo (HRP)")
            
            with st.spinner("Ejecutando simulaciones Monte Carlo para HRP..."):
                mc_var_hrp, mc_cvar_hrp, mc_returns_hrp = monte_carlo_var_cvar(
                    returns, hrp_weights, num_mc_sims
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("VaR 95% (Monte Carlo - HRP)", f"{mc_var_hrp*100:.2f}%")
                st.metric("CVaR 95% (Monte Carlo - HRP)", f"{mc_cvar_hrp*100:.2f}%")
            
            with col2:
                fig_mc_hrp = go.Figure()
                fig_mc_hrp.add_trace(go.Histogram(
                    x=mc_returns_hrp*100,
                    nbinsx=50,
                    name='Distribución de Retornos (HRP)',
                    marker_color='lightgreen'
                ))
                
                fig_mc_hrp.add_vline(x=mc_var_hrp*100, line_dash="dash", line_color="red",
                               annotation_text=f"VaR 95%: {mc_var_hrp*100:.2f}%")
                fig_mc_hrp.add_vline(x=mc_cvar_hrp*100, line_dash="dash", line_color="darkred",
                               annotation_text=f"CVaR 95%: {mc_cvar_hrp*100:.2f}%")
                
                fig_mc_hrp.update_layout(
                    title="Distribución de Retornos Monte Carlo (HRP)",
                    xaxis_title="Retorno (%)",
                    yaxis_title="Frecuencia",
                    height=400
                )
                
                st.plotly_chart(fig_mc_hrp, use_container_width=True)
            
            # Rendimiento histórico HRP
            st.subheader("📈 Rendimiento Histórico (HRP)")
            
            portfolio_returns_hrp = returns.dot(hrp_weights)
            cumulative_returns_hrp = (1 + portfolio_returns_hrp).cumprod()
            
            fig_hrp_perf = go.Figure()
            
            # Agregar ambos portafolios para comparar
            fig_hrp_perf.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode='lines',
                name='Markowitz',
                line=dict(color='rgb(0, 128, 255)', width=2)
            ))
            
            fig_hrp_perf.add_trace(go.Scatter(
                x=cumulative_returns_hrp.index,
                y=cumulative_returns_hrp.values,
                mode='lines',
                name='HRP',
                line=dict(color='rgb(0, 200, 100)', width=2)
            ))
            
            fig_hrp_perf.add_hline(y=1, line_dash="dash", line_color="gray", 
                         annotation_text="Inversión inicial")
            
            fig_hrp_perf.update_layout(
                title="Comparación de Rendimiento Histórico: Markowitz vs HRP",
                xaxis_title="Fecha",
                yaxis_title="Retorno Acumulado",
                height=400
            )
            
            st.plotly_chart(fig_hrp_perf, use_container_width=True)
            
            # Drawdown comparison
            st.subheader("📉 Comparación de Drawdowns")
            
            rolling_max_hrp = cumulative_returns_hrp.expanding().max()
            drawdowns_hrp = cumulative_returns_hrp/rolling_max_hrp - 1
            
            fig_dd_comp = go.Figure()
            
            fig_dd_comp.add_trace(go.Scatter(
                x=drawdowns.index,
                y=drawdowns.values,
                fill='tozeroy',
                name='Markowitz',
                line=dict(color='rgba(0, 128, 255, 0.5)')
            ))
            
            fig_dd_comp.add_trace(go.Scatter(
                x=drawdowns_hrp.index,
                y=drawdowns_hrp.values,
                fill='tozeroy',
                name='HRP',
                line=dict(color='rgba(0, 200, 100, 0.5)')
            ))
            
            fig_dd_comp.update_layout(
                title="Comparación de Drawdowns: Markowitz vs HRP",
                xaxis_title="Fecha",
                yaxis_title="Drawdown",
                height=400
            )
            
            st.plotly_chart(fig_dd_comp, use_container_width=True)
            
            # Descargar resultados como CSV
            st.markdown("---")
            st.markdown("## 💾 Descargar Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Preparar datos Markowitz para descargar
                download_data_markowitz = pd.DataFrame({
                    'Stock': stocks,
                    'Weight': optimal_weights,
                    'Amount': [w*initial_capital for w in optimal_weights]
                })
                
                csv_markowitz = download_data_markowitz.to_csv(index=False)
                b64_markowitz = base64.b64encode(csv_markowitz.encode()).decode()
                href_markowitz = f'<a href="data:file/csv;base64,{b64_markowitz}" download="optimal_portfolio_markowitz.csv">📥 Descargar Portfolio Markowitz (CSV)</a>'
                st.markdown(href_markowitz, unsafe_allow_html=True)
            
            with col2:
                # Preparar datos HRP para descargar
                download_data_hrp = pd.DataFrame({
                    'Stock': stocks,
                    'Weight': hrp_weights,
                    'Amount': [w*initial_capital for w in hrp_weights]
                })
                
                csv_hrp = download_data_hrp.to_csv(index=False)
                b64_hrp = base64.b64encode(csv_hrp.encode()).decode()
                href_hrp = f'<a href="data:file/csv;base64,{b64_hrp}" download="optimal_portfolio_hrp.csv">📥 Descargar Portfolio HRP (CSV)</a>'
                st.markdown(href_hrp, unsafe_allow_html=True)
            
            # Consejos de implementación
            st.markdown("---")
            st.markdown("## 💡 Consejos de Implementación")
            st.info("""
            **Basado en los principios de Portfolio Management:**
            
            1. **Rebalanceo**: Considera rebalancear tu portafolio trimestralmente para mantener los pesos óptimos
               - Usa disparadores automáticos: +30% sobreponderación o -20% infraponderación
            
            2. **Diversificación Real**: Este análisis incluye correlaciones y clustering jerárquico para una verdadera diversificación
            
            3. **Retorno Real**: Siempre considera el impacto de la inflación en tus rendimientos
               - Tu retorno real Markowitz: """ + f"{(results.iloc[optimal_idx]['Return'] - inflation_rate)*100:.2f}%" + """
               - Tu retorno real HRP: """ + f"{real_return_hrp*100:.2f}%" + """
            
            4. **Horizonte temporal**: Esta optimización es más efectiva para inversiones a mediano-largo plazo (15-20 años)
            
            5. **Gestión de Riesgo**: 
               - Mantén liquidez táctica (5% recomendado)
               - Monitorea el VaR y CVaR regularmente
               - Revisa el máximo drawdown que estás dispuesto a tolerar
            
            6. **Dos opiniones son mejor que una**: Compara los resultados de Markowitz y HRP para tomar mejores decisiones
            """)

# Agregar el footer
add_footer()