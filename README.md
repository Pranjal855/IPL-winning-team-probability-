The IPL Win Probability Predictor is an advanced machine learning application that forecasts real-time winning chances for competing teams during Indian Premier League matches.

Project Overview

• Objective: Build a dynamic, data-driven system to estimate live match win percentages based on evolving game states and historical data.
• Impact: Enhances real-time fan engagement, assists broadcast commentators, and offers tactical decision support for franchise analysts.

Technical Architecture & Methodology

• Data Pipeline: Uses historical ball-by-ball and match-level statistics from past IPL seasons (2008–present) processed via Python libraries like Pandas and NumPy.
• Feature Engineering: Extracts vital situational variables including current run rate (CRR), required run rate (RRR), runs remaining, wickets fallen, balls left, toss decisions, and venue specifics.
• Core Modeling: Implements classification algorithms such as Logistic Regression for direct probability mapping alongside ensemble methods like Random Forest and XGBoost for capturing complex, non-linear match dynamics.
• Deployment & Interface: Serializes the trained model using Pickle and integrates it into an interactive, user-friendly web dashboard built with Streamlit or Flask for instantaneous projections.
