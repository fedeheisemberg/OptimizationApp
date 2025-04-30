import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.stats import norm
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
    sharpe = (port_ret - rf_rate) / port_vol
    
    # Calcular VaR y CVaR
    port_returns = returns.dot(weights)
    var_95 = np.percentile(port_returns, 5)
    cvar_95 = port_returns[port_returns <= var_95].mean()
    
    # Maximum Drawdown
    cum_returns = (1 + port_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns/rolling_max - 1
    max_drawdown = drawdowns.min()
    
    return port_ret, port_vol, sharpe, var_95, cvar_95, max_drawdown

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
        💼 Made with ❤️ by Fede Martinez - Finanzas & Data
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# Título y descripción con emojis
st.title("📊 Advanced Portfolio Optimizer")
st.markdown("""
Esta aplicación te permite simular y optimizar portafolios de inversión utilizando la teoría moderna de portafolios.
Incluye costos de transacción, slippage y métricas avanzadas de riesgo para un análisis completo.

### ¿Cómo funciona? 🧠
1. Ingresa los símbolos de tus acciones preferidas
2. Ajusta los parámetros de inversión
3. Haz clic en "Run Optimization" para generar portafolios óptimos

### Conceptos clave 🔑
- **Ratio de Sharpe**: Mide el rendimiento ajustado por riesgo (mayor es mejor)
- **VaR**: Pérdida máxima esperada con 95% de confianza
- **CVaR**: Pérdida promedio esperada en los peores escenarios
- **Drawdown**: Caída desde máximos históricos
""")

# Sidebar - Parámetros de entrada
with st.sidebar:
    st.header("🛠️ Portfolio Parameters")
    
    # Input para tickers con explicación
    st.markdown("**Símbolos de acciones** 📈")
    default_tickers = "AAPL,MSFT,GOOGL,AMZN,JPM"
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
                                    value=100000,
                                    help="Monto total a invertir")
    
    rf_rate = st.slider("Tasa libre de riesgo (%) 🏦", 
                       min_value=0.0, 
                       max_value=10.0, 
                       value=4.0,
                       help="Rendimiento de bonos del tesoro o similar") / 100
    
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
    num_simulations = st.slider("Número de simulaciones", 
                              min_value=100, 
                              max_value=5000, 
                              value=1000,
                              help="Más simulaciones = mayor precisión pero más tiempo de ejecución")

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
            st.markdown("### 🔄 Simulación Monte Carlo")
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
            
            # Encontrar portafolio óptimo
            optimal_idx = results['Sharpe'].argmax()
            optimal_weights = all_weights[optimal_idx,:]
            
            st.success("✅ Optimización completada con éxito!")
            
            # Display results in multiple columns
            st.markdown("## 📊 Resultados del Portafolio Óptimo")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📈 Métricas del Portafolio Óptimo")
                metrics_df = pd.DataFrame({
                    'Métrica': ['Retorno Esperado', 'Volatilidad', 'Ratio de Sharpe',
                             'VaR (95%)', 'CVaR (95%)', 'Máximo Drawdown'],
                    'Valor': [f"{results.iloc[optimal_idx]['Return']*100:.2f}%",
                             f"{results.iloc[optimal_idx]['Volatility']*100:.2f}%",
                             f"{results.iloc[optimal_idx]['Sharpe']:.2f}",
                             f"{results.iloc[optimal_idx]['VaR_95']*100:.2f}%",
                             f"{results.iloc[optimal_idx]['CVaR_95']*100:.2f}%",
                             f"{results.iloc[optimal_idx]['Max_Drawdown']*100:.2f}%"]
                })
                st.dataframe(metrics_df)
                
                # Explicación de métricas
                with st.expander("ℹ️ Explicación de métricas"):
                    st.markdown("""
                    - **Retorno Esperado**: Rendimiento anualizado proyectado
                    - **Volatilidad**: Medida de riesgo (desviación estándar anualizada)
                    - **Ratio de Sharpe**: Retorno ajustado por riesgo (mayor = mejor)
                    - **VaR (95%)**: Pérdida máxima diaria con 95% de confianza
                    - **CVaR (95%)**: Pérdida promedio en el peor 5% de los escenarios
                    - **Máximo Drawdown**: Caída máxima desde un pico
                    """)
            
            with col2:
                st.subheader("💼 Pesos Óptimos")
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
                    title="Distribución del Portafolio"
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
            
            # Visualizaciones
            st.markdown("## 📊 Visualizaciones")
            
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
                name="Portafolio Óptimo"
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
                title="Rendimiento Histórico del Portafolio",
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
                title="Análisis de Drawdown del Portafolio",
                xaxis_title="Fecha",
                yaxis_title="Drawdown",
                height=400
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
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
            
            # Descargar resultados como CSV
            st.markdown("## 💾 Descargar Resultados")
            
            # Preparar datos para descargar
            download_data = pd.DataFrame({
                'Stock': stocks,
                'Weight': optimal_weights,
                'Amount': [w*initial_capital for w in optimal_weights]
            })
            
            csv = download_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="optimal_portfolio.csv">Descargar asignación óptima (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Consejos de implementación
            st.markdown("## 💡 Consejos de Implementación")
            st.info("""
            1. **Rebalanceo**: Considera rebalancear tu portafolio trimestralmente para mantener los pesos óptimos
            2. **Diversificación**: Más activos generalmente ofrecen mejor relación riesgo/retorno
            3. **Horizonte temporal**: Esta optimización es más efectiva para inversiones a mediano-largo plazo
            4. **Revisión periódica**: Reoptimiza tu portafolio cuando cambien las condiciones del mercado
            """)

# Agregar el footer
add_footer()