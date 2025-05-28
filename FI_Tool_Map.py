import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def calculate_years_to_fi(initial_portfolio, annual_expenses, annual_roi, swr, net_income):
    """Calculate the exact years needed to reach FI using compound interest and extend timeline after FI."""
    
    current_portfolio = initial_portfolio  # 🔥 Initialize current_portfolio

    annual_savings = net_income - annual_expenses
    fire_number = (annual_expenses / swr) * 100  # FIRE number based on SWR
    years = 0
    age = []
    portfolio_values = []
    cumulative_contributions = []
    cumulative_returns = []
    total_contributions = 0
    fire_year_exact = None


    # 🔥 Simulate Growth Until FIRE is Reached
    while current_portfolio < fire_number:
        age.append(age_input + years)
        portfolio_values.append(current_portfolio)
        cumulative_contributions.append(total_contributions)
        cumulative_returns.append(current_portfolio - total_contributions - initial_portfolio)

        current_portfolio += current_portfolio * (annual_roi / 100) + annual_savings
        total_contributions += annual_savings
        years += 1

    # 🔥 Compute Exact FIRE Year (Interpolation)
    for i in range(len(portfolio_values) - 1):
        if portfolio_values[i] < fire_number and portfolio_values[i + 1] >= fire_number:
            fire_year_exact = age_input + years + ((fire_number - portfolio_values[i]) / (portfolio_values[i + 1] - portfolio_values[i]) * (age[i + 1] - age[i]))

            # 🔥 Insert FIRE Age (43.4) into the data for hover accuracy
            age.insert(i + 1, fire_year_exact)
            portfolio_values.insert(i + 1, fire_number)
            cumulative_contributions.insert(i + 1, cumulative_contributions[i])
            cumulative_returns.insert(i + 1, cumulative_returns[i])
            break

    # 🔥 Extend Timeline by 3 Extra Years After FI
    for _ in range(3):
        age.append(age[-1] + 1)  # Continue age progression
        portfolio_values.append(current_portfolio)
        cumulative_contributions.append(total_contributions)
        cumulative_returns.append(current_portfolio - total_contributions - initial_portfolio)

        current_portfolio += current_portfolio * (annual_roi / 100) + annual_savings
        total_contributions += annual_savings

    return fire_year_exact, age, portfolio_values, cumulative_contributions, cumulative_returns, fire_number

# Streamlit UI
st.markdown("<h2 style='text-align: center;'> The tool runs with the following data input: </h2>", unsafe_allow_html=True)
#st.markdown("<h2 style='text-align: center;'>🔥 Financial Independence Calculator 🔥</h2>", unsafe_allow_html=True)
#st.markdown(
#    "<p style='text-align: center; color: grey; font-size: 0.95em;'>📱 On mobile? Tap Fullscreen mode below and, after entering your data, rotate your phone horizontally to view the chart and map properly.</p>",
#    unsafe_allow_html=True
#)


# Instructions Section (Collapsible)
#with st.expander("📌 **Instructions** (Click to expand/collapse)", expanded=False):
#    st.write("""
#    - This tool helps you calculate your **timeline for reaching financial independence (FI)**, when you can live off your investment portfolio.
#    - The tool calculates 1) what your **target portfolio** should be (your **FI number**), based on your retirement spending needs, and 2) the **number of years** it will likely take you to get there based on average returns.
#    - The tool also estimates 3) how your FI timeline changes **if you decide to retire abroad**. To do so, it considers data on cost of living (Numbeo, 2025). The map and table **compare FI timelines across 106 countries**. 
#    - Input the following **key financial details**: Net Annual Income (after tax); Current Annual Expenses; Current Portfolio Value; Expected Annual (Real) Return on Investment (%); Safe Withdrawal Rate (%) (FAQs below); Projected Annual Expenses in Retirement; Current Country of Residence; Current Age.
#    - You can enter the values in your country currency.
#    - The tool assumes you **invest the difference** between your Net Annual Income and Current Annual Expenses.
#    - You can **download the plot and map** as .png and the table as .csv.
#    """)



# Load Cost of Living Data
file_path = "https://raw.githubusercontent.com/Gadamer007/FI_Calculator/main/Col_Sal.xlsx"
xls = pd.ExcelFile(file_path)
df_col = pd.read_excel(xls, sheet_name="Country", usecols=["Country", "Col"])  # Read Country and COL


# User Inputs
# 🔥 Create a more compact input layout with 4 rows and 2 columns
# User Inputs (Create a more compact input layout with 4 rows and 2 columns)
col1, col2 = st.columns(2)

# In col1, place the first set of inputs
with col1:
    net_income = st.number_input("Net Annual Income ($)", min_value=0, value=85000, step=1000)
    current_portfolio = st.number_input("Current Portfolio Value ($)", min_value=0, value=200000, step=10000)
    swr = st.number_input("Safe Withdrawal Rate (%)", min_value=1.0, max_value=10.0, value=4.0, step=0.1)
    selected_country = st.selectbox("Select Your Base Country", df_col["Country"].unique(), key="base_country_select_1")

# In col2, place the second set of inputs
with col2:
    annual_expenses = st.number_input("Current Annual Expenses ($)", min_value=0, value=41000, step=1000)
    annual_roi = st.number_input("Annual Return on Investment (%)", min_value=0.0, max_value=20.0, value=7.0, step=0.1)
    retirement_expenses = st.number_input("Estimated Annual Expenses in Retirement ($)", min_value=0, value=annual_expenses, step=1000)
    age_input = st.number_input("Enter your current age", min_value=1, max_value=100, value=30, step=1)

# Then, you can use the user input variables (age_input, selected_country, etc.) in your calculations.



# 🔥 Calculate and Display Savings Rate in a Green Box
savings_rate = ((net_income - annual_expenses) / net_income) * 100



df_col.columns = ["Country", "COL_Index"]  # Rename columns for easier use
df_col.dropna(inplace=True)  # Drop any NaN values



# Get COL of Selected Country
selected_col_index = df_col.loc[df_col["Country"] == selected_country, "COL_Index"].values[0]


# Calculate Years to FI and Display Chart
years_to_fi, age, portfolio_values, cumulative_contributions, cumulative_returns, fire_number = calculate_years_to_fi(
    current_portfolio, retirement_expenses, annual_roi, swr, net_income)

# 🔥 **Find Exact FIRE Year (Interpolation)**
fire_year_exact = None
years_until_fi = None  # 🔥 Ensure this variable is defined

for i in range(len(portfolio_values) - 1):
    if portfolio_values[i] < fire_number and portfolio_values[i + 1] >= fire_number:
        fire_year_exact = age[i] + (fire_number - portfolio_values[i]) / (portfolio_values[i + 1] - portfolio_values[i]) * (age[i + 1] - age[i])
        years_until_fi = fire_year_exact - age_input  # 🔥 Compute years until FI from starting age (age_input)
        break

st.success(
    f"💰 Your current Savings Rate is **{savings_rate:.1f}%**."
    f"\n\n🏆 You will reach Financial Independence in **{years_until_fi:.1f} years** (age **{fire_year_exact:.1f}**)."
    f"\n\n🎯 Your target portfolio value is **${fire_number:,.0f}**."
)
#st.success(f"💰 Your current Savings Rate is **{savings_rate:.1f}%**")

# **Softer & Modern Color Palette**
colors = {
    "Initial Portfolio": "rgba(120, 144, 156, 0.8)",  # Soft Blue-Grey
    "Cumulative Contributions": "rgba(255, 193, 7, 0.7)",  # Soft Yellow-Gold
    "Cumulative Returns": "rgba(76, 175, 80, 0.7)",  # Soft Green
    "Total Net Worth": "rgba(41, 182, 246, 1)",  # Sky Blue
}

# **Stacking Corrected**
fig = go.Figure()

# **Initial Portfolio**
fig.add_trace(go.Scatter(
    x=age, y=[current_portfolio] * len(age),  # 🔥 Now updates dynamically
    fill='tozeroy', mode='none', name="Initial Portfolio",
    fillcolor=colors["Initial Portfolio"]
))

# **Cumulative Contributions**
contributions_cumulative = np.array(cumulative_contributions) + current_portfolio  # 🔥 Use dynamically updated portfolio

fig.add_trace(go.Scatter(
    x=age, y=contributions_cumulative,
    fill='tonexty', mode='none', name="Contributions",  # 🔥 Renamed
    fillcolor=colors["Cumulative Contributions"]
))

fig.add_trace(go.Scatter(
    x=age, y=np.array(cumulative_returns) + contributions_cumulative,  # 🔥 Corrected variable
    fill='tonexty', mode='none', name="Returns",
    fillcolor=colors["Cumulative Returns"]
))

# **Total Net Worth Line**
fig.add_trace(go.Scatter(
    x=age, y=portfolio_values,
    mode='lines', name="Total Net Worth",
    line=dict(color=colors["Total Net Worth"], width=3)
))

# **FIRE Threshold Line**
fig.add_trace(go.Scatter(
    x=age, y=[fire_number] * len(age),
    mode='lines', name="FIRE Number",
    line=dict(color='red', dash='dash')
))

# **Light Grey Vertical Dashed Line (Hidden from Legend)**
if fire_year_exact is not None:
    fig.add_trace(go.Scatter(
        x=[fire_year_exact, fire_year_exact], y=[0, fire_number],
        mode='lines', line=dict(color='lightgrey', dash='dash'),
        showlegend=False
    ))

    # 🔥 Find the closest index to fire_year_exact in the age list
    lower_index = max(i for i in range(len(age)) if age[i] <= fire_year_exact)
    upper_index = min(i for i in range(len(age)) if age[i] >= fire_year_exact)

    # 🔥 Linearly interpolate contributions, returns, and net worth at FIRE age
    weight_upper = (fire_year_exact - age[lower_index]) / (age[upper_index] - age[lower_index])
    weight_lower = 1 - weight_upper

    interpolated_contributions = (
        cumulative_contributions[lower_index] * weight_lower + cumulative_contributions[upper_index] * weight_upper
    )
    interpolated_returns = (
        cumulative_returns[lower_index] * weight_lower + cumulative_returns[upper_index] * weight_upper
    )
    interpolated_net_worth = (
        portfolio_values[lower_index] * weight_lower + portfolio_values[upper_index] * weight_upper
    )

    fire_hover_text = (
        f"<b>Age:</b> {fire_year_exact:.1f}<br>"
        f"<b>Initial Portfolio:</b> ${current_portfolio:,.0f}<br>"
        f"<b>Cumulative Contributions:</b> ${interpolated_contributions:,.0f}<br>"
        f"<b>Cumulative Returns:</b> ${interpolated_returns:,.0f}<br>"
        f"<b>Total Net Worth:</b> ${interpolated_net_worth:,.0f}"
    )


    fig.add_trace(go.Scatter(
        x=[fire_year_exact], y=[fire_number],
        mode='markers',
        marker=dict(color='red', size=10),
        name="FIRE Marker",
        hoverinfo="text",  # 🔥 Enables hover text
        text=[fire_hover_text],  # 🔥 Uses the same format as other points
        showlegend=False
    ))

    # **Move Annotation Even Higher to Fully Clear the Red Bubble**
    fig.add_annotation(
        x=fire_year_exact,
        y=fire_number + (fire_number * 0.12),  # 🔥 Move it even higher
        text=f"{years_until_fi:.1f} years<br>(age {fire_year_exact:.1f})",
        showarrow=False,
        font=dict(size=14, color="white"),
        align="center"
    )

# **X and Y Axis Styling**
fig.update_xaxes(
    showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.3)", zeroline=True, zerolinecolor="white",
    color="white", title_text="Age", title_font=dict(size=14, color="white"),
    tickfont=dict(color="white"),  # 🔥 Make X-axis values white
    showline=True, linecolor="white",
    range=[min(age), max(age)]
)
fig.update_yaxes(
    showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.3)", zeroline=True, zerolinecolor="white",
    color="white", title_text="Portfolio Value ($)", title_font=dict(size=14, color="white"),
    tickfont=dict(color="white"),  # 🔥 Make Y-axis values white
    showline=True, linecolor="white",
    range=[0, max(portfolio_values) * 1.1]
)

# **Final Layout Adjustments**
fig.update_layout(
    title=dict(
        text="Road to Financial Independence",
        x=0.5,  # 🔥 Center the title
        xanchor="center",  # 🔥 Force centering
        yanchor="top",
        font=dict(size=20, color="white")
    ),
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    legend=dict(font=dict(color='white')),
    showlegend=True
)

# 🔥 Create Custom Hover Labels for Each Age
hover_texts = [
    f"<b>Age:</b> {a}<br>"
    f"<b>Initial Portfolio:</b> $530,000<br>"
    f"<b>Cumulative Contributions:</b> ${c:,.0f}<br>"
    f"<b>Cumulative Returns:</b> ${r:,.0f}<br>"
    f"<b>Total Net Worth:</b> ${p:,.0f}"
    for a, c, r, p in zip(age, cumulative_contributions, cumulative_returns, portfolio_values)
]

# 🔥 Apply Custom Hover Text to All Traces (EXCEPT FIRE MARKER)
for trace in fig.data:
    if trace.name and "FIRE" not in trace.name:  # 🔥 Check if name exists before filtering
        trace.hoverinfo = "text"
        trace.text = hover_texts

# 🔥 Render Final Chart
st.plotly_chart(fig)


import streamlit as st
import numpy as np
import pandas as pd

def calculate_years_to_fi(initial_portfolio, annual_expenses, annual_roi, swr, net_income):
    """Compute the years needed to reach Financial Independence."""
    current_portfolio = initial_portfolio
    annual_savings = net_income - annual_expenses
    fire_number = (annual_expenses / swr) * 100
    years = 0

    while current_portfolio < fire_number:
        current_portfolio += current_portfolio * (annual_roi / 100) + annual_savings
        years += 1

    # Interpolation for fractional year
    last_portfolio = current_portfolio - (current_portfolio * (annual_roi / 100) + annual_savings)
    fire_year_exact = years - 1 + ((fire_number - last_portfolio) / (current_portfolio - last_portfolio))

    return round(fire_year_exact, 1)  # Rounded to 1 decimal place

# Load Cost of Living Data
file_path = "https://raw.githubusercontent.com/Gadamer007/FI_Calculator/main/Col_Sal.xlsx"
xls = pd.ExcelFile(file_path)
df_col = pd.read_excel(xls, sheet_name="Country", usecols=["Country", "Col"])
df_col.columns = ["Country", "COL_Index"]
df_col.dropna(inplace=True)  # Drop NaN values

# Get COL index for selected country
#selected_country2 = st.selectbox("🌍 Select Your Base Country", df_col["Country"].unique(), key="base_country_select_2")

selected_col_index = df_col.loc[df_col["Country"] == selected_country, "COL_Index"].values[0]

# Compute relative COL and adjusted retirement expenses
df_col["Relative COL (%)"] = (df_col["COL_Index"] / selected_col_index) * 100
df_col["Adjusted Retirement Expenses ($)"] = (df_col["Relative COL (%)"] / 100) * retirement_expenses

# Compute Updated FI Timeline for each country
df_col["Updated FI Timeline (Years)"] = df_col["Adjusted Retirement Expenses ($)"].apply(
    lambda x: calculate_years_to_fi(current_portfolio, x, annual_roi, swr, net_income)
)

# Display updated table
#st.markdown(f"### 🌎 Cost of Living Comparison (Relative to {selected_country})")
#st.dataframe(df_col.sort_values(by="Updated FI Timeline (Years)", ascending=False))


# new
import plotly.express as px

# Convert negative values to 0 for display purposes
df_col["Display FI Timeline"] = df_col["Updated FI Timeline (Years)"].apply(lambda x: max(x, 0))

# Define Custom Color Scale (Dark Green for FI Achieved, Gradient for others)
custom_colorscale = [
    (0.0, "darkgreen"),  # 🟢 FI Achieved (0 years)
    (0.2, "lightgreen"), # 🟢 Almost FI (~0.2 years)
    (0.5, "yellow"),     # 🟡 Mid-range
    (0.8, "orange"),     # 🟠 Approaching FI
    (1.0, "darkred")     # 🔴 Still far from FI
]

# Modify Hover Text (Show "Years to FI: 0" for negatives)
df_col["Hover Text"] = df_col.apply(
    lambda row: f"{row['Country']}<br>{row['Updated FI Timeline (Years)']:.1f} years" 
    if row["Updated FI Timeline (Years)"] >= 0 else f"{row['Country']}<br>0 years",
    axis=1
)


# Create Choropleth Map
fig_map = px.choropleth(
    df_col,
    locations="Country",
    locationmode="country names",
    color="Display FI Timeline",
    color_continuous_scale=custom_colorscale,
    hover_name=None,  # Remove the first bold country name
    hover_data={"Country": True, "Display FI Timeline": True},  # Keep country name + years to FI
    title="🌍 FI Timeline relocating to other countries (Map)",
    labels={"Display FI Timeline": "Years to FI"},
)




# Improve Layout
fig_map.update_layout(
    geo=dict(showcoastlines=True, projection_type="natural earth"),
    margin={"r":0,"t":90,"l":0,"b":40},  # Increase space above and below
    coloraxis_colorbar=dict(title="Years to FI"),
    title_x=0.15  # Center title
)


# Count countries where FI Timeline is under 1 year
num_fi_countries = (df_col["Updated FI Timeline (Years)"] <= 0.1).sum()


# Display in Streamlit
st.plotly_chart(fig_map)

st.success(
    f"\n\n🌍 You could already retire in **{num_fi_countries} out of 106** countries in our dataset (see map and table)"
)

st.dataframe(
    df_col[["Country", "Relative COL (%)", "Adjusted Retirement Expenses ($)", "Display FI Timeline"]]
    .rename(columns={"Display FI Timeline": "FI Timeline"})
    .sort_values(by="FI Timeline", ascending=False)
    .round(1)  # Round all numeric values to 1 decimal place
    .reset_index(drop=True)  # 🔥 This removes the index safely
)

