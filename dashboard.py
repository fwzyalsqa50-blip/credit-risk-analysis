import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(r"D:\Data analyst\Credit_Risk\UCI_Credit_Card.csv")
df.rename(columns={'default.payment.next.month': 'default'}, inplace=True)
df.drop('ID', axis=1, inplace=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_clients   = len(df)
total_defaults  = df['default'].sum()
default_rate    = df['default'].mean() * 100
avg_limit       = df['LIMIT_BAL'].mean()
losses_prevented = 796 * 50000

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Credit Risk Dashboard"

TEAL  = "#028090"
CORAL = "#FF6B6B"
GOLD  = "#F59E0B"
WHITE = "#FFFFFF"
DARK  = "#1E293B"

def kpi_card(title, value, color, icon):
    return dbc.Card([
        dbc.CardBody([
            html.Div(icon, style={"fontSize": "2rem", "marginBottom": "8px"}),
            html.H2(value, style={"color": color, "fontWeight": "bold", "margin": "0"}),
            html.P(title, style={"color": "#94A3B8", "margin": "0", "fontSize": "0.85rem"}),
        ], style={"textAlign": "center", "padding": "20px"})
    ], style={"background": DARK, "border": f"1px solid {color}", "borderRadius": "12px"})

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Credit Risk Analytics Dashboard",
                    style={"color": TEAL, "fontWeight": "bold", "marginBottom": "4px"}),
            html.P("UCI Credit Card Dataset  |  30,000 Clients  |  Polynomial Logistic Regression",
                   style={"color": "#64748B", "fontSize": "0.9rem"})
        ])
    ], style={"padding": "24px 0 12px 0"}),

    html.Hr(style={"borderColor": TEAL, "marginBottom": "24px"}),

    # KPI Cards
    dbc.Row([
        dbc.Col(kpi_card("Total Clients",     f"{total_clients:,}",          TEAL,  "👥"), width=3),
        dbc.Col(kpi_card("Total Defaults",    f"{total_defaults:,}",         CORAL, "⚠️"), width=3),
        dbc.Col(kpi_card("Default Rate",      f"{default_rate:.1f}%",        GOLD,  "📊"), width=3),
        dbc.Col(kpi_card("Avg Credit Limit",  f"NT${avg_limit:,.0f}",        TEAL,  "💳"), width=3),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(kpi_card("Model AUC",         "0.7551",                      TEAL,  "🎯"), width=3),
        dbc.Col(kpi_card("Model Accuracy",    "75.03%",                      GOLD,  "✅"), width=3),
        dbc.Col(kpi_card("Detection Rate",    "60.0%",                       CORAL, "🔍"), width=3),
        dbc.Col(kpi_card("Losses Prevented",  f"NT${losses_prevented:,}",    TEAL,  "💰"), width=3),
    ], className="mb-4"),

    html.Hr(style={"borderColor": "#334155", "marginBottom": "24px"}),

    # Row 1: Default Distribution + Age Distribution
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Default Distribution",
                               style={"color": WHITE, "background": DARK,
                                      "borderBottom": f"2px solid {TEAL}"}),
                dbc.CardBody([dcc.Graph(id="default-dist")])
            ], style={"background": DARK, "border": "none"})
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Age Distribution by Default Status",
                               style={"color": WHITE, "background": DARK,
                                      "borderBottom": f"2px solid {TEAL}"}),
                dbc.CardBody([dcc.Graph(id="age-dist")])
            ], style={"background": DARK, "border": "none"})
        ], width=6),
    ], className="mb-4"),

    # Row 2: Credit Limit + Payment Status
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Credit Limit Distribution by Default",
                               style={"color": WHITE, "background": DARK,
                                      "borderBottom": f"2px solid {GOLD}"}),
                dbc.CardBody([dcc.Graph(id="limit-dist")])
            ], style={"background": DARK, "border": "none"})
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Default Rate by Payment Status (PAY_0)",
                               style={"color": WHITE, "background": DARK,
                                      "borderBottom": f"2px solid {GOLD}"}),
                dbc.CardBody([dcc.Graph(id="pay-status")])
            ], style={"background": DARK, "border": "none"})
        ], width=6),
    ], className="mb-4"),

    # Row 3: Education + Marriage
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Default Rate by Education Level",
                               style={"color": WHITE, "background": DARK,
                                      "borderBottom": f"2px solid {CORAL}"}),
                dbc.CardBody([dcc.Graph(id="edu-chart")])
            ], style={"background": DARK, "border": "none"})
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Default Rate by Marital Status",
                               style={"color": WHITE, "background": DARK,
                                      "borderBottom": f"2px solid {CORAL}"}),
                dbc.CardBody([dcc.Graph(id="marriage-chart")])
            ], style={"background": DARK, "border": "none"})
        ], width=6),
    ], className="mb-4"),

    # Row 4: Confusion Matrix + ROC
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Confusion Matrix",
                               style={"color": WHITE, "background": DARK,
                                      "borderBottom": f"2px solid {TEAL}"}),
                dbc.CardBody([dcc.Graph(id="confusion-matrix")])
            ], style={"background": DARK, "border": "none"})
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Interactive Risk Assessment",
                               style={"color": WHITE, "background": DARK,
                                      "borderBottom": f"2px solid {GOLD}"}),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Payment Delay (PAY_0)", style={"color": "#94A3B8"}),
                            dcc.Slider(id="pay-slider", min=-2, max=8, step=1, value=0,
                                      marks={i: str(i) for i in range(-2, 9)},
                                      tooltip={"placement": "bottom"}),
                            html.Br(),
                            html.Label("Credit Limit (NT$)", style={"color": "#94A3B8"}),
                            dcc.Slider(id="limit-slider", min=10000, max=500000,
                                      step=10000, value=100000,
                                      tooltip={"placement": "bottom"}),
                            html.Br(),
                            html.Label("Age", style={"color": "#94A3B8"}),
                            dcc.Slider(id="age-slider", min=20, max=80,
                                      step=1, value=35,
                                      tooltip={"placement": "bottom"}),
                        ])
                    ]),
                    html.Br(),
                    html.Div(id="risk-output",
                             style={"textAlign": "center", "padding": "20px",
                                    "borderRadius": "8px", "background": "#0F172A"})
                ])
            ], style={"background": DARK, "border": "none"})
        ], width=6),
    ], className="mb-4"),

    # Footer
    html.Hr(style={"borderColor": "#334155"}),
    html.P("Credit Risk Analytics  |  Polynomial Logistic Regression  |  2026",
           style={"color": "#475569", "textAlign": "center", "fontSize": "0.8rem",
                  "paddingBottom": "16px"})

], fluid=True, style={"background": "#0F172A", "minHeight": "100vh"})

# ── Callbacks ─────────────────────────────────────────────────────────────────
PLOT_BG = "#1E293B"
PAPER_BG = "#1E293B"
FONT_COLOR = "#E2E8F0"

def base_layout():
    return dict(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR, family="Calibri"),
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )

@app.callback(Output("default-dist", "figure"), Input("default-dist", "id"))
def default_dist(_):
    counts = df['default'].value_counts().reset_index()
    counts.columns = ['Status', 'Count']
    counts['Status'] = counts['Status'].map({0: 'No Default', 1: 'Default'})
    fig = px.bar(counts, x='Status', y='Count',
                 color='Status', color_discrete_map={'No Default': TEAL, 'Default': CORAL},
                 text='Count')
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig.update_layout(**base_layout(), showlegend=False)
    return fig

@app.callback(Output("age-dist", "figure"), Input("age-dist", "id"))
def age_dist(_):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[df['default']==0]['AGE'], name='No Default',
                               opacity=0.7, marker_color=TEAL, nbinsx=25))
    fig.add_trace(go.Histogram(x=df[df['default']==1]['AGE'], name='Default',
                               opacity=0.7, marker_color=CORAL, nbinsx=25))
    fig.update_layout(**base_layout(), barmode='overlay',
                      xaxis_title="Age", yaxis_title="Count")
    return fig

@app.callback(Output("limit-dist", "figure"), Input("limit-dist", "id"))
def limit_dist(_):
    fig = go.Figure()
    fig.add_trace(go.Box(y=df[df['default']==0]['LIMIT_BAL'], name='No Default',
                         marker_color=TEAL, boxmean=True))
    fig.add_trace(go.Box(y=df[df['default']==1]['LIMIT_BAL'], name='Default',
                         marker_color=CORAL, boxmean=True))
    fig.update_layout(**base_layout(), yaxis_title="Credit Limit (NT$)")
    return fig

@app.callback(Output("pay-status", "figure"), Input("pay-status", "id"))
def pay_status(_):
    pay_df = df.groupby('PAY_0')['default'].mean().reset_index()
    pay_df['default'] = pay_df['default'] * 100
    fig = px.bar(pay_df, x='PAY_0', y='default',
                 color='default', color_continuous_scale=['#028090', '#FF6B6B'],
                 text=pay_df['default'].round(1).astype(str) + '%')
    fig.update_traces(textposition='outside')
    fig.update_layout(**base_layout(), xaxis_title="Payment Status",
                      yaxis_title="Default Rate %", coloraxis_showscale=False)
    return fig

@app.callback(Output("edu-chart", "figure"), Input("edu-chart", "id"))
def edu_chart(_):
    edu_map = {1: 'Graduate', 2: 'University', 3: 'High School', 4: 'Others'}
    edu_df = df[df['EDUCATION'].isin([1,2,3,4])].copy()
    edu_df['EDU_NAME'] = edu_df['EDUCATION'].map(edu_map)
    edu_agg = edu_df.groupby('EDU_NAME')['default'].mean().reset_index()
    edu_agg['default'] = edu_agg['default'] * 100
    fig = px.bar(edu_agg, x='EDU_NAME', y='default',
                 color='default', color_continuous_scale=['#028090', '#FF6B6B'],
                 text=edu_agg['default'].round(1).astype(str) + '%')
    fig.update_traces(textposition='outside')
    fig.update_layout(**base_layout(), xaxis_title="Education",
                      yaxis_title="Default Rate %", coloraxis_showscale=False)
    return fig

@app.callback(Output("marriage-chart", "figure"), Input("marriage-chart", "id"))
def marriage_chart(_):
    mar_map = {0: 'Unknown', 1: 'Married', 2: 'Single', 3: 'Other'}
    mar_df = df.copy()
    mar_df['MAR_NAME'] = mar_df['MARRIAGE'].map(mar_map)
    mar_agg = mar_df.groupby('MAR_NAME')['default'].mean().reset_index()
    mar_agg['default'] = mar_agg['default'] * 100
    fig = px.bar(mar_agg, x='MAR_NAME', y='default',
                 color='default', color_continuous_scale=['#028090', '#FF6B6B'],
                 text=mar_agg['default'].round(1).astype(str) + '%')
    fig.update_traces(textposition='outside')
    fig.update_layout(**base_layout(), xaxis_title="Marital Status",
                      yaxis_title="Default Rate %", coloraxis_showscale=False)
    return fig

@app.callback(Output("confusion-matrix", "figure"), Input("confusion-matrix", "id"))
def confusion_matrix_chart(_):
    z = [[3706, 967], [531, 796]]
    x = ['Predicted: No Default', 'Predicted: Default']
    y = ['Actual: No Default', 'Actual: Default']
    text = [['3,706\nTrue Negative', '967\nFalse Positive'],
            ['531\nFalse Negative', '796\nTrue Positive']]
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, text=text,
                               texttemplate="%{text}",
                               colorscale=[[0, '#0F172A'], [1, TEAL]],
                               showscale=False))
    fig.update_layout(**base_layout())
    return fig

@app.callback(
    Output("risk-output", "children"),
    Input("pay-slider", "value"),
    Input("limit-slider", "value"),
    Input("age-slider", "value"),
)
def risk_assessment(pay, limit, age):
    base_prob = 0.10
    if pay >= 2:   base_prob += 0.45
    elif pay == 1: base_prob += 0.25
    elif pay == 0: base_prob += 0.10
    if limit < 50000:  base_prob += 0.15
    elif limit < 150000: base_prob += 0.05
    if age > 50: base_prob += 0.05
    base_prob = min(base_prob, 0.97)

    if base_prob < 0.30:
        risk, color, action = "LOW RISK",      TEAL,  "Approve credit increase"
        emoji = "✅"
    elif base_prob < 0.60:
        risk, color, action = "MODERATE RISK", GOLD,  "Monitor account closely"
        emoji = "⚠️"
    else:
        risk, color, action = "HIGH RISK",     CORAL, "Reduce limit / Flag for review"
        emoji = "🚨"

    return [
        html.Div(emoji, style={"fontSize": "2.5rem"}),
        html.H2(f"{base_prob:.0%}", style={"color": color, "fontWeight": "bold", "margin": "8px 0"}),
        html.H4("Default Probability", style={"color": "#94A3B8", "margin": "0"}),
        html.Hr(style={"borderColor": color, "margin": "12px 0"}),
        html.H3(risk, style={"color": color, "fontWeight": "bold", "margin": "0"}),
        html.P(action, style={"color": "#CBD5E1", "margin": "8px 0 0 0"})
    ]

if __name__ == "__main__":
    app.run(debug=True)