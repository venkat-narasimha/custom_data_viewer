import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Configuration ---
st.set_page_config(layout="wide", page_title="FO OI Trend Dashboard")

# --- Load Data ---
@st.cache_data # Cache the data loading to improve performance
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['TradDt'])
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        # Identify numeric columns (OI change columns + Price)
        oi_cols = ['0FUT', 'CE,ATM', 'CE,I 1-5', 'CE,I >5', 'CE,O 1-5', 'CE,O >5',
                   'PE,ATM', 'PE,I 1-5', 'PE,I >5', 'PE,O 1-5', 'PE,O >5']
        # Convert OI columns to numeric, coercing errors (blanks become NaN)
        for col in oi_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill missing OI values with 0 (assuming blank means no change recorded)
        df[oi_cols] = df[oi_cols].fillna(0).astype(int)
        # Ensure Price is numeric
        if 'UndrlygPric' in df.columns:
             df['UndrlygPric'] = pd.to_numeric(df['UndrlygPric'], errors='coerce')

        df.sort_values('TradDt', inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None

DATA_FILE = 'monthly_oi_new_series_summary - Trend_all_FO.csv'
df = load_data(DATA_FILE)

if df is None:
    st.stop() # Stop execution if data loading failed

# --- Sidebar Controls ---
st.sidebar.header("Filters")

# Stock Selector
all_stocks = sorted(df['TckrSymb'].unique())
selected_stocks = st.sidebar.multiselect(
    "Select Stock(s)",
    options=all_stocks,
    default=all_stocks[0] if all_stocks else [] # Default to first stock if available
)

# Date Range Selector
min_date = df['TradDt'].min().date()
max_date = df['TradDt'].max().date()

start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

# --- Filter Data ---
if not selected_stocks:
    st.warning("Please select at least one stock.")
    st.stop()

if start_date > end_date:
    st.warning("Start date cannot be after end date.")
    st.stop()

# Convert start_date and end_date back to Timestamps for comparison
start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)

mask = (
    (df['TckrSymb'].isin(selected_stocks)) &
    (df['TradDt'] >= start_ts) &
    (df['TradDt'] <= end_ts)
)
filtered_df = df[mask].copy() # Use .copy()

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# --- Calculations for KPIs and Charts ---
# Define Call and Put columns accurately based on cleaned names
ce_cols = ['CE,ATM', 'CE,I 1-5', 'CE,I >5', 'CE,O 1-5', 'CE,O >5']
pe_cols = ['PE,ATM', 'PE,I 1-5', 'PE,I >5', 'PE,O 1-5', 'PE,O >5']

# Ensure only existing columns are used for summation
valid_ce_cols = [col for col in ce_cols if col in filtered_df.columns]
valid_pe_cols = [col for col in pe_cols if col in filtered_df.columns]

filtered_df['Total CE Change'] = filtered_df[valid_ce_cols].sum(axis=1)
filtered_df['Total PE Change'] = filtered_df[valid_pe_cols].sum(axis=1)

# Calculate PCR based on OI Change (handle division by zero)
# Replace 0 in denominator with NaN temporarily, calculate, then fill NaN/inf with 0
denominator = filtered_df['Total CE Change'].replace(0, np.nan)
filtered_df['PCR (OI Change)'] = (filtered_df['Total PE Change'] / denominator).fillna(0).replace([np.inf, -np.inf], 0)


# --- Dashboard Display ---
st.title(f"Futures & Options OI Trends: {', '.join(selected_stocks)}")

# --- KPIs ---
st.subheader("Summary Metrics (Selected Period)")
col1, col2, col3, col4 = st.columns(4)

# Get latest data point for selected stocks within the date range
latest_data = filtered_df.loc[filtered_df.groupby('TckrSymb')['TradDt'].idxmax()]

with col1:
    if len(selected_stocks) == 1 and not latest_data.empty:
        st.metric("Latest Price", f"₹{latest_data['UndrlygPric'].iloc[0]:,.2f}", f"{latest_data['TradDt'].iloc[0].strftime('%Y-%m-%d')}")
    else:
        st.metric("Latest Price", "N/A (Multiple Stocks)")

with col2:
    total_fut_oi = filtered_df['0FUT'].sum()
    st.metric("Total Futures OI Chg", f"{total_fut_oi:,.0f}")

with col3:
    total_ce_oi = filtered_df['Total CE Change'].sum()
    st.metric("Total Call OI Chg", f"{total_ce_oi:,.0f}")

with col4:
    total_pe_oi = filtered_df['Total PE Change'].sum()
    st.metric("Total Put OI Chg", f"{total_pe_oi:,.0f}")

st.markdown("---") # Separator

# --- Charts ---
st.subheader("Trend Charts")

# Underlying Price
if 'UndrlygPric' in filtered_df.columns:
    fig_price = px.line(filtered_df, x='TradDt', y='UndrlygPric', color='TckrSymb',
                        title='Underlying Price Trend', labels={'UndrlygPric': 'Price (₹)'})
    fig_price.update_layout(hovermode="x unified")
    st.plotly_chart(fig_price, use_container_width=True)

# Futures OI Change
if '0FUT' in filtered_df.columns:
    fig_fut = px.line(filtered_df, x='TradDt', y='0FUT', color='TckrSymb',
                      title='Futures OI Change Trend', labels={'0FUT': 'Futures OI Change'})
    fig_fut.update_layout(hovermode="x unified")
    st.plotly_chart(fig_fut, use_container_width=True)

# Total Call vs Put OI Change
fig_total_oi = px.line(filtered_df, x='TradDt', y=['Total CE Change', 'Total PE Change'], color='TckrSymb',
                       title='Total Call vs Put OI Change Trend', labels={'value': 'Total OI Change'})
fig_total_oi.update_layout(hovermode="x unified")
st.plotly_chart(fig_total_oi, use_container_width=True)

# PCR Trend
fig_pcr = px.line(filtered_df, x='TradDt', y='PCR (OI Change)', color='TckrSymb',
                  title='PCR (OI Change) Trend', labels={'PCR (OI Change)': 'Put-Call Ratio (OI Change)'})
fig_pcr.update_layout(hovermode="x unified")
st.plotly_chart(fig_pcr, use_container_width=True)


# OI Breakdowns (Example for Calls - you can add Puts similarly)
st.subheader("OI Change Breakdown")
col_b1, col_b2 = st.columns(2)

with col_b1:
    if valid_ce_cols:
        fig_ce_breakdown = px.line(filtered_df, x='TradDt', y=valid_ce_cols, color='TckrSymb',
                                   title='Call OI Change Breakdown by Type', labels={'value': 'OI Change'})
        fig_ce_breakdown.update_layout(hovermode="x unified")
        st.plotly_chart(fig_ce_breakdown, use_container_width=True)
    else:
        st.write("No Call OI data columns found.")

with col_b2:
    if valid_pe_cols:
        fig_pe_breakdown = px.line(filtered_df, x='TradDt', y=valid_pe_cols, color='TckrSymb',
                                   title='Put OI Change Breakdown by Type', labels={'value': 'OI Change'})
        fig_pe_breakdown.update_layout(hovermode="x unified")
        st.plotly_chart(fig_pe_breakdown, use_container_width=True)
    else:
        st.write("No Put OI data columns found.")


st.markdown("---") # Separator

# --- Data Table ---
st.subheader("Filtered Data")
# Display relevant columns, format numbers maybe
display_cols = ['TckrSymb', 'TradDt', 'UndrlygPric', '0FUT', 'Total CE Change', 'Total PE Change', 'PCR (OI Change)'] + valid_ce_cols + valid_pe_cols
st.dataframe(filtered_df[display_cols].round(2)) # Show the filtered dataframe

# --- Optional: Download Button ---
@st.cache_data
def convert_df_to_csv(df_to_convert):
   return df_to_convert.to_csv(index=False).encode('utf-8')

csv_download = convert_df_to_csv(filtered_df[display_cols])

st.download_button(
   label="Download Filtered Data as CSV",
   data=csv_download,
   file_name=f'filtered_oi_data_{"_".join(selected_stocks)}_{start_date}_to_{end_date}.csv',
   mime='text/csv',
)