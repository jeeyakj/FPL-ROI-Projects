import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

@st.cache_data
def load_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()

    players = pd.DataFrame(data['elements'])
    teams = pd.DataFrame(data['teams'])
    positions = pd.DataFrame(data['element_types'])

    team_map = dict(zip(teams['id'], teams['name']))
    position_map = dict(zip(positions['id'], positions['singular_name']))

    players['Team'] = players['team'].map(team_map)
    players['Position'] = players['element_type'].map(position_map)
    players['Name'] = players['first_name'] + ' ' + players['second_name']
    players['Cost (£M)'] = players['now_cost'] / 10
    players['ROI'] = players['total_points'] / players['Cost (£M)']

    return players[['Name', 'Team', 'Position', 'Cost (£M)', 'total_points', 'ROI']]

df = load_fpl_data()


st.set_page_config(page_title="FPL ROI Calculator", layout="wide")
st.title("Fantasy Premier League ROI Calculator")
st.markdown("Compare player value based on points per £1M")


positions = ['All'] + sorted(df['Position'].unique())
selected_position = st.selectbox("Filter by Position", positions)

if selected_position != 'All':
    filtered_df = df[df['Position'] == selected_position]
else:
    filtered_df = df


st.subheader("Top ROI Players")
st.dataframe(filtered_df.sort_values(by='ROI', ascending=False).reset_index(drop=True))


top_roi = filtered_df.sort_values(by='ROI', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_roi['Name'], top_roi['ROI'], color='pink')
ax.set_xlabel('ROI (Points per £1M)')
ax.set_title('Top 10 ROI Players')
ax.invert_yaxis()
st.pyplot(fig)

#python -m streamlit run fpl_roi_app.py
