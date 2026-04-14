"""Reusable Plotly chart builders for factor analysis.

Each function accepts tool outputs (DataFrame / dict) and returns a
``plotly.graph_objects.Figure``. Charts do NO computation — tools compute,
charts visualize.

Usage in scripts::

    fig = cumulative_ic_chart(ic_series)
    fig.show()

Usage in Streamlit::

    st.plotly_chart(cumulative_ic_chart(ic_series))
"""
