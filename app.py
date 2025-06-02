from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

app = Flask(__name__)

# Load COL data once at startup
COL_SHEET_URL = (
    "https://raw.githubusercontent.com/Gadamer007/FI_Calculator/"
    "main/Col_Sal.xlsx"
)
df_col = pd.read_excel(COL_SHEET_URL, sheet_name="Country", usecols=["Country", "Col"])
df_col.columns = ["Country", "COL_Index"]
df_col.dropna(inplace=True)

@app.route("/")
def index():
    countries = df_col["Country"].tolist()
    return render_template("index.html", countries=countries)

@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.get_json()
    # parse inputs
    initial_portfolio = float(data["initialPortfolio"])
    annual_expenses   = float(data["annualExpenses"])
    net_income        = float(data["netIncome"])
    roi               = float(data["roi"])
    swr               = float(data["swr"])
    retire_exp        = float(data["retirementExpenses"])
    age_input         = int(data["age"])
    country           = data["country"]

    # FIRE target and savings
    fire_number    = (annual_expenses / swr) * 100
    annual_savings = net_income - annual_expenses

    # build timeline until FIRE + 3 years
    ages, port_vals, contribs, returns_ = [], [], [], []
    total_contrib = 0.0
    cp = initial_portfolio
    year = 0
    while cp < fire_number:
        ages.append(age_input + year)
        port_vals.append(cp)
        contribs.append(total_contrib)
        returns_.append(cp - total_contrib - initial_portfolio)
        cp = cp * (1 + roi/100) + annual_savings
        total_contrib += annual_savings
        year += 1
    # extend 3 more
    for _ in range(3):
        ages.append(ages[-1] + 1)
        port_vals.append(cp)
        contribs.append(total_contrib)
        returns_.append(cp - total_contrib - initial_portfolio)
        cp = cp * (1 + roi/100) + annual_savings
        total_contrib += annual_savings

    # interpolate exact FIRE age
    fire_year_exact = None
    years_until_fi  = None
    for i in range(len(port_vals)-1):
        if port_vals[i] < fire_number <= port_vals[i+1]:
            frac = (fire_number - port_vals[i])/(port_vals[i+1]-port_vals[i])
            fire_year_exact = ages[i] + frac*(ages[i+1]-ages[i])
            years_until_fi  = fire_year_exact - age_input
            break

    # build stacked arrays
    layer0 = np.full(len(ages), initial_portfolio)
    layer1 = layer0 + np.array(contribs)
    layer2 = layer1 + np.array(returns_)

    # ——————————————————————————————————————————————————————
    # build hover‐text arrays (rounded, correct deltas)
    hover_initial  = [f"${initial_portfolio:,.0f}"] * len(ages)
    hover_contribs = [f"${c:,.0f}"           for c in contribs]
    hover_returns  = [f"${r:,.0f}"           for r in returns_]
    hover_net      = [f"${v:,.0f}"           for v in port_vals]
    
    # build plotly figure
    fig = go.Figure()
    
    # 1) Initial Portfolio
    fig.add_trace(go.Scatter(
        x=ages, y=layer0,
        fill="tozeroy", mode="none",
        name="Initial Portfolio",
        fillcolor="rgba(120,144,156,0.8)",
        hoverinfo="text", text=hover_initial
    ))
    
    # 2) Contributions
    fig.add_trace(go.Scatter(
        x=ages, y=layer1,
        fill="tonexty", mode="none",
        name="Contributions",
        fillcolor="rgba(255,193,7,0.6)",
        hoverinfo="text", text=hover_contribs
    ))
    
    # 3) Returns
    fig.add_trace(go.Scatter(
        x=ages, y=layer2,
        fill="tonexty", mode="none",
        name="Returns",
        fillcolor="rgba(76,175,80,0.6)",
        hoverinfo="text", text=hover_returns
    ))
    
    # 4) Total Net Worth (line on top)
    fig.add_trace(go.Scatter(
        x=ages, y=port_vals,
        mode="lines", name="Total Net Worth",
        line=dict(color="deepskyblue", width=3),
        hoverinfo="text", text=hover_net
    ))
    
    # 5) FIRE Number line (hover only shows FIRE value, not age)
    fig.add_trace(go.Scatter(
        x=ages, y=[fire_number] * len(ages),
        mode="lines", name="FIRE Number",
        line=dict(color="red", dash="dash"),
        hovertemplate="$%{y:.0f}<extra></extra>"
    ))
    
    # FIRE marker & annotation (hover only shows FIRE value)
    if fire_year_exact is not None:
        fig.add_shape(dict(
            type="line", x0=fire_year_exact, x1=fire_year_exact,
            y0=0,           y1=fire_number,
            line=dict(color="lightgrey", dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=[fire_year_exact], y=[fire_number],
            mode="markers", showlegend=False,
            marker=dict(color="red", size=10),
            hovertemplate="$%{y:.0f}<extra></extra>"
        ))
        fig.add_annotation(
            x=fire_year_exact, y=fire_number * 1.05,
            text=f"{years_until_fi:.1f} yrs (age {fire_year_exact:.1f})",
            font=dict(color="white"), showarrow=False
        )

    # axes styling (ensure Y-axis line is shown, legend below, “compare-on-hover” active)
    fig.update_layout(
        title=dict(text="<b>Road to Financial Independence</b>", x=0.5),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        hovermode="x unified",
        margin=dict(b=40),  # give some bottom margin for the horizontal legend

        # ─── Legend BELOW the chart ───
        legend=dict(
            orientation="h",       # horizontal legend
            yanchor="top",
            y=-0.2,                # push it below the plot
            xanchor="center",
            x=0.5,
            font=dict(color="white")
        ),

        xaxis=dict(
            title="Age",
            titlefont=dict(color='white', size=14),
            tickfont=dict(color='white'),
            color='white',
            gridcolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            title="Portfolio Value ($)",
            titlefont=dict(color='white', size=14),
            tickfont=dict(color='white'),
            color='white',
            showline=True,
            linecolor='white',
            gridcolor='rgba(255,255,255,0.2)'
        )
    )

    # ——————————————————————————————————————————————————————
    # build map + table
    base_idx = df_col.loc[df_col.Country == country, "COL_Index"].iloc[0]
    df = df_col.copy()
    df["Relative COL (%)"] = df.COL_Index / base_idx * 100
    df["Adj Ret Exp ($)"]  = df["Relative COL (%)"] / 100 * retire_exp

    def ft(c_exp):
        cp2, yrs2 = initial_portfolio, 0
        tgt = c_exp / swr * 100
        while cp2 < tgt and yrs2 < 100:
            cp2 = cp2 * (1 + roi / 100) + annual_savings
            yrs2 += 1
        return yrs2 - 1 + (tgt - (cp2 - cp2 * (roi / 100) - annual_savings)) / (
            (cp2) - (cp2 - cp2 * (roi / 100) - annual_savings)
        )

    df["FI Timeline"] = df["Adj Ret Exp ($)"].apply(ft).clip(lower=0).round(1)

    fig_map = px.choropleth(
        df,
        locations="Country",
        locationmode="country names",
        color="FI Timeline",
        color_continuous_scale=[
            (0, "darkgreen"),
            (0.2, "lightgreen"),
            (0.5, "yellow"),
            (0.8, "orange"),
            (1, "darkred"),
        ]
    )
    
    fig_map.update_layout(
        title=dict(
            text="🌍 FI Timeline When Relocating Abroad<br><span style='font-size:0.8em'>(Click on countries below)</span>",
            x=0.5,
            xanchor="center"
        ),
        margin=dict(r=0, t=90, l=0, b=40),
        hovermode="x unified",
        coloraxis_showscale=False
    )






    # Force the single PX trace’s colorbar to be horizontal:
    #fig_map.data[0].colorbar.orientation = "h"
    #fig_map.data[0].colorbar.x = 0.5
    #fig_map.data[0].colorbar.y = 1.05
    #fig_map.data[0].colorbar.xanchor = "center"
    #fig_map.data[0].colorbar.yanchor = "bottom"

    return jsonify(
        portfolioChart=fig.to_json(),
        mapChart=fig_map.to_json(),
        fireYear=round(fire_year_exact, 1) if fire_year_exact else None,
        yearsUntilFI=round(years_until_fi, 1) if years_until_fi else None,
        fireNumber=round(fire_number, 1),
        fiReadyCount=int((df["FI Timeline"] <= 0.1).sum()),
        # -- Include these two values so JS can compute savings rate
        netIncome=net_income,
        annualExpenses=annual_expenses,
        countryTable=(
            df[["Country", "Relative COL (%)", "Adj Ret Exp ($)", "FI Timeline"]]
            .assign(**{
                "Relative COL (%)": lambda d: d["Relative COL (%)"].round(1),
                "Adj Ret Exp ($)":  lambda d: d["Adj Ret Exp ($)"].round(0),
            })
            .rename(columns={
                "Adj Ret Exp ($)": "Adjusted Retirement Expenses ($)",
                "FI Timeline": "Display FI Timeline"
            })
            .to_dict("records")
        )
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 5000)), debug=False)

