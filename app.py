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

    fire_year_exact = None
    years_until_fi = None
    for i in range(len(portfolio_values) - 1):
        if portfolio_values[i] < fire_number and portfolio_values[i + 1] >= fire_number:
            fire_year_exact = age[i] + ((fire_number - portfolio_values[i]) / (portfolio_values[i + 1] - portfolio_values[i])) * (age[i + 1] - age[i])
            years_until_fi = fire_year_exact - age_input

            # Insert interpolated point for FIRE
            age.insert(i + 1, fire_year_exact)
            portfolio_values.insert(i + 1, fire_number)
            cumulative_contributions.insert(i + 1, cumulative_contributions[i])
            cumulative_returns.insert(i + 1, cumulative_returns[i])
            break

    colors = {
        "Initial Portfolio": "rgba(120, 144, 156, 0.8)",
        "Cumulative Contributions": "rgba(255, 193, 7, 0.7)",
        "Cumulative Returns": "rgba(76, 175, 80, 0.7)",
        "Total Net Worth": "rgba(41, 182, 246, 1)",
    }

    fig = go.Figure()

    # Initial Portfolio
    initial_layer = np.array([initial_portfolio] * len(age))
    fig.add_trace(go.Scatter(x=age, y=initial_layer, fill='tozeroy', mode='none', name="Initial Portfolio", fillcolor=colors["Initial Portfolio"]))

    # Cumulative Contributions
    contributions_cumulative = np.array(cumulative_contributions) + initial_layer
    fig.add_trace(go.Scatter(x=age, y=contributions_cumulative, fill='tonexty', mode='none', name="Contributions", fillcolor=colors["Cumulative Contributions"]))

    # Cumulative Returns
    returns_cumulative = np.array(cumulative_returns) + contributions_cumulative
    fig.add_trace(go.Scatter(x=age, y=returns_cumulative, fill='tonexty', mode='none', name="Returns", fillcolor=colors["Cumulative Returns"]))

    # Total Net Worth Line
    fig.add_trace(go.Scatter(x=age, y=portfolio_values, mode='lines', name="Total Net Worth", line=dict(color=colors["Total Net Worth"], width=3)))

    # FIRE Number Line
    fig.add_trace(go.Scatter(x=age, y=[fire_number] * len(age), mode='lines', name="FIRE Number", line=dict(color='red', dash='dash')))

    if fire_year_exact is not None:
        fig.add_trace(go.Scatter(x=[fire_year_exact, fire_year_exact], y=[0, fire_number], mode='lines', line=dict(color='lightgrey', dash='dash'), showlegend=False))
        fig.add_trace(go.Scatter(x=[fire_year_exact], y=[fire_number], mode='markers', marker=dict(color='red', size=10), name="FIRE Marker", showlegend=False))
        fig.add_annotation(x=fire_year_exact, y=fire_number + (fire_number * 0.12), text=f"{years_until_fi:.1f} years<br>(age {fire_year_exact:.1f})", showarrow=False, font=dict(size=14, color="white"), align="center")

    fig.update_layout(title=dict(text="Road to Financial Independence", x=0.5, xanchor="center", yanchor="top", font=dict(size=20, color="white")), plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'), legend=dict(font=dict(color='white')), showlegend=True)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.3)", zeroline=True, zerolinecolor="white", color="white", title_text="Age", title_font=dict(size=14, color="white"), tickfont=dict(color="white"), showline=True, linecolor="white", range=[min(age), max(age)])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.3)", zeroline=True, zerolinecolor="white", color="white", title_text="Portfolio Value ($)", title_font=dict(size=14, color="white"), tickfont=dict(color="white"), showline=True, linecolor="white", range=[0, max(portfolio_values) * 1.1])

    selected_col_index = df_col.loc[df_col["Country"] == selected_country, "COL_Index"].values[0]
    df = df_col.copy()
    df["Relative COL (%)"] = (df["COL_Index"] / selected_col_index) * 100
    df["Adjusted Retirement Expenses ($)"] = (df["Relative COL (%)"] / 100) * retirement_expenses

    def calc_fi_timeline(country_exp):
        cp = initial_portfolio
        savings = net_income - country_exp
        target = (country_exp / swr) * 100
        years = 0
        while cp < target and years < 100:
            cp += cp * (roi / 100) + savings
            years += 1
        return years - 1 + ((target - (cp - cp * (roi / 100) - savings)) / ((cp) - (cp - cp * (roi / 100) - savings)))

    df["Updated FI Timeline (Years)"] = df["Adjusted Retirement Expenses ($)"].apply(calc_fi_timeline)
    df["Display FI Timeline"] = df["Updated FI Timeline (Years)"].apply(lambda x: max(x, 0))

    fig_map = px.choropleth(df, locations="Country", locationmode="country names", color="Display FI Timeline", hover_name="Country", color_continuous_scale=[(0.0, "darkgreen"), (0.2, "lightgreen"), (0.5, "yellow"), (0.8, "orange"), (1.0, "darkred")], title="🌍 FI Timeline relocating to other countries (Map)", labels={"Display FI Timeline": "Years to FI"})
    fig_map.update_layout(geo=dict(showcoastlines=True, projection_type="natural earth"), margin={"r":0,"t":90,"l":0,"b":40}, coloraxis_colorbar=dict(title="Years to FI"), title_x=0.15)

    fi_ready_count = (df["Updated FI Timeline (Years)"] <= 0.1).sum()
    country_table = df[["Country", "Relative COL (%)", "Adjusted Retirement Expenses ($)", "Display FI Timeline"]]
    country_table = country_table.sort_values(by="Display FI Timeline", ascending=False).round(1).to_dict(orient='records')

    return jsonify({
        'portfolioChart': fig.to_json(),
        'mapChart': fig_map.to_json(),
        'fireYear': round(fire_year_exact, 1),
        'yearsUntilFI': round(years_until_fi, 1),
        'fireNumber': round(fire_number, 1),
        'fiReadyCount': int(fi_ready_count),
        'countryTable': country_table
    })




