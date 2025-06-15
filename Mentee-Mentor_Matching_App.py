import streamlit as st

st.title("Mentee-Mentor Matching App")

interest = st.selectbox('Select Interest Field', mentee_df['Mentee_InterestField'].unique())
location = st.selectbox('Select Location', mentee_df['Mentee_Location'].unique())
burnout = st.slider('Burnout Risk Score', 0.0, 10.0, 3.0)

if st.button('Find Mentors'):
    cluster, mentor_list = recommend_mentors({
        'InterestField': interest,
        'Location': location,
        'BurnoutRiskScore': burnout
    })
    st.write(f'Mentee belongs to cluster {cluster}')
    st.write('Top recommended mentors:')
    for mentor in mentor_list:
        st.write(f"Mentor ID: {mentor['Mentor_ID']}, Field: {mentor['Mentor_InterestField']}, Location: {mentor['Mentor_Location']}, Similarity: {mentor['Similarity']:.4f}")
