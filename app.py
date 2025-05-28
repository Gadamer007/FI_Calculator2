from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

app = Flask(__name__)

# Load Excel data
xls_url = "https://raw.githubusercontent.com/Gadamer007/FI_Calculator/main/Col_Sal.xlsx"
df_col = pd.read_excel(xls_url, sheet_name="Country", usecols=["Country", "Col"])
df_col.columns = ["Country", "COL_Index"]
df_col.dropna(inplace=True)

@app.route('/')
def index():
    countries = df_col["Country"].unique().tolist()
    return render_template("index.html", countries=countries)

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    initial_portfolio = float(data['initialPortfolio'])
    annual_expenses = float(data['annualExpenses'])
    net_income = float(data['netIncome'])
    roi = float(data['roi'])
    swr = float(data['swr'])
    retirement_expenses = float(data['retirementExpenses'])
    age_input = int(data['age'])
    selected_country = data['country']

    # --- Core FI Calculation ---
    current_portfolio = initial_portfolio
    fire_number = (annual_expenses / swr) * 100
    annual_savings = net_income - annual_expenses
    age = []
    portfolio_values = []
    cumulative_contributions = []
    cumulative_returns = []
    total_contributions = 0
    years = 0

    while current_portfolio < fire_number:
        age.append(age_input + years)
        portfolio_values.append(current_portfolio)
        cumulative_contributions.append(total_contributions)
        cumulative_returns.append(current_portfolio - total_contributions - initial_portfolio)
        current_portfolio += current_portfolio * (roi / 100) + annual_savings
        total_contributions += annual_savings
        years += 1

    for _ in range(3):
        age.append(age[-1] + 1)
        portfolio_values.append(current_portfolio)
        cumulative_contributions.append(total_contributions)
        cumulative_returns.append(current_portfolio - total_contributions - initial_portfolio)
        current_portfolio += current_portfolio * (roi / 100) + annual_savings
        total_contributions += annual_savings

    fire_year_exact = age_input + years
    years_until_fi = fire_year_exact - age_input

    # --- Portfolio Chart ---
    contributions_cumulative = np.array(cumulative_contributions) + initial_portfolio
    returns_cumulative = np.array(cumulative_returns) + contributions_cumulative

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=age, y=[initial_portfolio]*len(age), fill='tozeroy', mode='none', name="Initial Portfolio"))
    fig.add_trace(go.Scatter(x=age, y=contributions_cumulative, fill='tonexty', mode='none', name="Contributions"))
    fig.add_trace(go.Scatter(x=age, y=returns_cumulative, fill='tonexty', mode='none', name="Returns"))
    fig.add_trace(go.Scatter(x=age, y=portfolio_values, mode='lines', name='Total Net Worth'))
    fig.add_trace(go.Scatter(x=age, y=[fire_number]*len(age), mode='lines', name='FIRE Number', line=dict(dash='dash', color='red')))

    fig.update_layout(title='Portfolio Over Time', xaxis_title='Age', yaxis_title='Portfolio Value ($)')
    portfolio_chart_json = fig.to_json()

    # --- Country Comparison Map ---
    try:
        selected_col_index = df_col.loc[df_col["Country"] == selected_country, "COL_Index"].values[0]
    except IndexError:
        return jsonify({'error': 'Selected country not found in dataset'}), 400

    df = df_col.copy()
    df["Relative COL (%)"] = (df["COL_Index"] / selected_col_index) * 100
    df["Adjusted Retirement Expenses ($)"] = (df["Relative COL (%)"] / 100) * retirement_expenses

    def calculate_years(country_exp):
        cp = initial_portfolio
        savings = net_income - country_exp
        target = (country_exp / swr) * 100
        years = 0
        while cp < target and years < 100:
            cp += cp * (roi / 100) + savings
            years += 1
        return round(years, 1)

    df["Updated FI Timeline (Years)"] = df["Adjusted Retirement Expenses ($)"].apply(calculate_years)
    df["Display FI Timeline"] = df["Updated FI Timeline (Years)"].apply(lambda x: max(x, 0))

    fig_map = px.choropleth(
        df,
        locations="Country",
        locationmode="country names",
        color="Display FI Timeline",
        hover_name="Country",
        color_continuous_scale="YlGnBu",
        title="FI Timeline by Country"
    )
    fig_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    map_chart_json = fig_map.to_json()

    fi_ready_count = (df["Updated FI Timeline (Years)"] <= 0.1).sum()
    country_table = df[["Country", "Relative COL (%)", "Adjusted Retirement Expenses ($)", "Display FI Timeline"]]
    country_table = country_table.sort_values(by="Display FI Timeline", ascending=False).round(1).to_dict(orient='records')

    return jsonify({
        'portfolioChart': portfolio_chart_json,
        'mapChart': map_chart_json,
        'fireYear': round(fire_year_exact, 1),
        'yearsUntilFI': round(years_until_fi, 1),
        'fireNumber': round(fire_number),
        'fiReadyCount': int(fi_ready_count),
        'countryTable': country_table
    })

