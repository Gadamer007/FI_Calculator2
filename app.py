from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

app = Flask(__name__)

# Load Excel data once at startup
xls_url = "https://raw.githubusercontent.com/Gadamer007/FI_Calculator/main/Col_Sal.xlsx"
df_col = pd.read_excel(xls_url, sheet_name="Country", usecols=["Country", "Col"])
df_col.rename(columns={"Col": "COL_Index"}, inplace=True)
df_col.dropna(inplace=True)

@app.route('/')
def index():
    countries = df_col["Country"].tolist()
    return render_template("index.html", countries=countries)

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    initial_portfolio = float(data['initialPortfolio'])
    annual_expenses    = float(data['annualExpenses'])
    net_income         = float(data['netIncome'])
    roi                = float(data['roi'])
    swr                = float(data['swr'])
    retirement_exp     = float(data['retirementExpenses'])
    age_input          = int(data['age'])
    selected_country   = data['country']

    # FIRE number and annual savings
    fire_number    = (annual_expenses / swr) * 100
    annual_savings = net_income - annual_expenses

    # Build time series until FIRE + 3 more years
    ages                   = []
    portfolio_values       = []
    cumulative_contrib     = []
    cumulative_returns     = []
    total_contrib          = 0.0
    current_portfolio      = initial_portfolio
    year_idx               = 0

    # simulate up to FIRE
    while current_portfolio < fire_number:
        ages.append(age_input + year_idx)
        portfolio_values.append(current_portfolio)
        cumulative_contrib.append(total_contrib)
        cumulative_returns.append(current_portfolio - total_contrib - initial_portfolio)

        # compound + save
        current_portfolio = current_portfolio * (1 + roi/100) + annual_savings
        total_contrib += annual_savings
        year_idx += 1

    # extend 3 additional years
    for _ in range(3):
        ages.append(ages[-1] + 1)
        portfolio_values.append(current_portfolio)
        cumulative_contrib.append(total_contrib)
        cumulative_returns.append(current_portfolio - total_contrib - initial_portfolio)
        current_portfolio = current_portfolio * (1 + roi/100) + annual_savings
        total_contrib += annual_savings

    # interpolate exact FIRE age
    fire_year_exact = None
    years_until_fi  = None
    for i in range(len(portfolio_values)-1):
        if portfolio_values[i] < fire_number <= portfolio_values[i+1]:
            frac = (fire_number - portfolio_values[i]) / (portfolio_values[i+1] - portfolio_values[i])
            fire_year_exact = ages[i] + frac*(ages[i+1]-ages[i])
            years_until_fi  = fire_year_exact - age_input
            break

    # prepare stacked arrays
    initial_layer      = np.full(len(ages), initial_portfolio)
    contributions_layer= initial_layer + np.array(cumulative_contrib)
    returns_layer      = contributions_layer + np.array(cumulative_returns)

    # build figure
    fig = go.Figure()

    # 1) Initial Portfolio
    fig.add_trace(go.Scatter(
        x=ages, y=initial_layer,
        fill='tozeroy', mode='none',
        name='Initial Portfolio',
        fillcolor='rgba(120,144,156,0.8)'
    ))

    # 2) Contributions
    fig.add_trace(go.Scatter(
        x=ages, y=contributions_layer,
        fill='tonexty', mode='none',
        name='Contributions',
        fillcolor='rgba(255,193,7,0.6)'
    ))

    # 3) Returns
    fig.add_trace(go.Scatter(
        x=ages, y=returns_layer,
        fill='tonexty', mode='none',
        name='Returns',
        fillcolor='rgba(76,175,80,0.6)'
    ))

    # 4) Total Net Worth line
    fig.add_trace(go.Scatter(
        x=ages, y=portfolio_values,
        mode='lines', name='Total Net Worth',
        line=dict(color='deepskyblue', width=3)
    ))

    # 5) FIRE threshold line
    fig.add_trace(go.Scatter(
        x=ages, y=[fire_number]*len(ages),
        mode='lines', name='FIRE Number',
        line=dict(color='red', dash='dash')
    ))

    # vertical marker & annotation at FIRE
    if fire_year_exact is not None:
        fig.add_trace(go.Scatter(
            x=[fire_year_exact, fire_year_exact], y=[0, fire_number],
            mode='lines', showlegend=False,
            line=dict(color='lightgrey', dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=[fire_year_exact], y=[fire_number],
            mode='markers', showlegend=False,
            marker=dict(color='red', size=10)
        ))
        fig.add_annotation(
            x=fire_year_exact, y=fire_number * 1.05,
            text=f"{years_until_fi:.1f} yrs (age {fire_year_exact:.1f})",
            showarrow=False,
            font=dict(color='white')
        )

    # layout
    fig.update_layout(
        title=dict(text="Road to Financial Independence", x=0.5),
        plot_bgcolor='black', paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(title="Age", color='white', gridcolor='rgba(255,255,255,0.2)'),
        yaxis=dict(title="Portfolio Value ($)", color='white', gridcolor='rgba(255,255,255,0.2)')
    )

    # ——————————————————————————————————————————————————————
    # now build the map exactly as before...
    selected_col_index = df_col.loc[df_col.Country==selected_country, "COL_Index"].iloc[0]
    df = df_col.copy()
    df["Relative COL (%)"] = df.COL_Index / selected_col_index * 100
    df["Adjusted Retirement Expenses ($)"] = df["Relative COL (%)"]/100 * retirement_exp

    def calc_fi_timeline(country_exp):
        cp      = initial_portfolio
        savings = net_income - country_exp
        target  = country_exp / swr * 100
        yrs     = 0
        while cp < target and yrs<100:
            cp += cp*(roi/100) + savings
            yrs += 1
        return yrs - 1 + (target - (cp - cp*(roi/100) - savings)) / ((cp) - (cp - cp*(roi/100) - savings))

    df["Updated FI Timeline (Years)"] = df["Adjusted Retirement Expenses ($)"].apply(calc_fi_timeline)
    df["Display FI Timeline"] = df["Updated FI Timeline (Years)"].clip(lower=0)

    fig_map = px.choropleth(
        df, locations="Country", locationmode="country names",
        color="Display FI Timeline", hover_name="Country",
        color_continuous_scale=[
            (0.0,"darkgreen"),(0.2,"lightgreen"),
            (0.5,"yellow"),(0.8,"orange"),(1.0,"darkred")
        ],
        title="🌍 FI Timeline relocating to other countries (Map)",
        labels={"Display FI Timeline":"Years to FI"}
    )
    fig_map.update_layout(
        geo=dict(showcoastlines=True, projection_type="natural earth"),
        margin=dict(r=0,t=90,l=0,b=40),
        coloraxis_colorbar=dict(title="Years to FI"),
        title_x=0.15
    )

    fi_ready_count = int((df["Updated FI Timeline (Years)"]<=0.1).sum())
    country_table = df[["Country","Relative COL (%)","Adjusted Retirement Expenses ($)","Display FI Timeline"]]
    country_table = country_table.sort_values("Display FI Timeline", ascending=False).round(1).to_dict("records")

    return jsonify(
        portfolioChart=fig.to_json(),
        mapChart=fig_map.to_json(),
        fireYear=round(fire_year_exact,1) if fire_year_exact else None,
        yearsUntilFI=round(years_until_fi,1) if years_until_fi else None,
        fireNumber=round(fire_number,1),
        fiReadyCount=fi_ready_count,
        countryTable=country_table
    )

if __name__ == '__main__':
    app.run(debug=True)







