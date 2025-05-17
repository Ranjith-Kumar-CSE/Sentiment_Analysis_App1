import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, mean_squared_error
import seaborn as sns
import io

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# [Other functions like read_posts, decode_emotion, analyze_posts, etc., remain unchanged]

# Main app
def main():
    st.title("📊 Sentiment Analysis App")
    st.markdown("""
    Use the tabs below to analyze a single text post or a CSV file containing social media posts.
    Adjust sentiment thresholds as needed. The app uses VADER to analyze sentiments and display results.
    Download results and visualizations as needed.
    """)

    # Initialize session state variables
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = {}

    # Create tabs
    text_tab, csv_tab = st.tabs(["Analyze Text", "Analyze CSV File"])

    # Text Analysis Tab
    with text_tab:
        st.header("Analyze a Single Text Post")
        st.markdown("Enter a text post and adjust thresholds to analyze its sentiment.")

        # Inputs
        single_text = st.text_area("Enter a Text Post", height=100)
        pos_threshold_text = st.slider("Positive Sentiment Threshold", 0.0, 1.0, 0.5, 0.05, key="text_pos")
        neg_threshold_text = st.slider("Negative Sentiment Threshold", -1.0, 0.0, -0.5, 0.05, key="text_neg")
        analyze_text_button = st.button("Analyze Text")

        # Handle text analysis
        if analyze_text_button:
            if single_text.strip():
                with st.spinner("Analyzing text..."):
                    st.subheader("Text Analysis Result")
                    result = analyze_single_text(single_text, pos_threshold_text, neg_threshold_text)
                    st.text_area("Text Sentiment", result, height=100)
                    # Download text result
                    st.download_button(
                        label="Download Text Analysis Result",
                        data=result,
                        file_name="text_analysis_result.txt",
                        mime="text/plain"
                    )
            else:
                st.error("Please enter some text to analyze.")

    # CSV Analysis Tab
    with csv_tab:
        st.header("Analyze CSV File")
        st.markdown("Upload a CSV file, specify the text column, and adjust thresholds to analyze sentiments.")

        # Inputs
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        text_column = st.text_input("Text Column Name", value="text")
        pos_threshold_csv = st.slider("Positive Sentiment Threshold", 0.0, 1.0, 0.5, 0.05, key="csv_pos")
        neg_threshold_csv = st.slider("Negative Sentiment Threshold", -1.0, 0.0, -0.5, 0.05, key="csv_neg")
        analyze_button = st.button("Analyze CSV")

        # Handle CSV analysis
        if analyze_button and uploaded_file is not None:
            with st.spinner("Analyzing posts..."):
                posts, true_labels, df = read_posts(uploaded_file, text_column, pos_threshold_csv, neg_threshold_csv)
                
                if posts is not None and true_labels is not None:
                    # Analyze posts
                    predicted_labels, compound_scores, polarity_scores, results_text, analysis_data = analyze_posts(
                        posts, pos_threshold_csv, neg_threshold_csv
                    )

                    # Compute results for visualizations
                    results = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
                    for label in predicted_labels:
                        results[label] += 1

                    # Compute metrics
                    accuracy, f1, rmse, cm, fpr, tpr, roc_auc = compute_metrics(
                        true_labels, predicted_labels, compound_scores
                    )

                    # Store results in session state
                    st.session_state.analysis_results = {
                        'results_text': results_text,
                        'analysis_data': analysis_data,
                        'accuracy': accuracy,
                        'f1': f1,
                        'rmse': rmse,
                        'cm': cm,
                        'fpr': fpr,
                        'tpr': tpr,
                        'roc_auc': roc_auc,
                        'results': results,
                        'polarity_scores': polarity_scores
                    }

                    # Generate and store visualizations
                    st.session_state.visualizations = {
                        'pie_fig': create_pie_chart(results),
                        'bar_fig': create_bar_graph(results),
                        'cm_fig': create_confusion_matrix(cm),
                        'roc_fig': create_roc_curve(fpr, tpr, roc_auc),
                        'corr_fig': create_correlation_matrix(polarity_scores)
                    }

        # Display results if they exist in session state
        if st.session_state.analysis_results is not None:
            st.subheader("CSV Analysis Results")
            
            # Sentiment analysis output
            with st.expander("Sentiment Analysis Output", expanded=False):
                st.text_area("Post Sentiments", st.session_state.analysis_results['results_text'], height=300)
            # Download sentiment analysis results
            analysis_df = pd.DataFrame(st.session_state.analysis_results['analysis_data'])
            csv_buffer = io.StringIO()
            analysis_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Sentiment Analysis Results",
                data=csv_buffer.getvalue(),
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )

            # Metrics
            st.subheader("Evaluation Metrics (Auto-Generated Labels)")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{st.session_state.analysis_results['accuracy']:.3f}")
                st.metric("F1 Score (Weighted)", f"{st.session_state.analysis_results['f1']:.3f}")
            with col2:
                st.metric("RMSE", f"{st.session_state.analysis_results['rmse']:.3f}")
                st.metric("ROC AUC", f"{st.session_state.analysis_results['roc_auc']:.3f}")
            
            # Download metrics report
            metrics_report = f"""Sentiment Analysis Metrics Report
Accuracy: {st.session_state.analysis_results['accuracy']:.3f}
F1 Score (Weighted): {st.session_state.analysis_results['f1']:.3f}
RMSE: {st.session_state.analysis_results['rmse']:.3f}
ROC AUC: {st.session_state.analysis_results['roc_auc']:.3f}
"""
            st.download_button(
                label="Download Metrics Report",
                data=metrics_report,
                file_name="metrics_report.txt",
                mime="text/plain"
            )

            st.write("**Confusion Matrix (Rows: True, Columns: Predicted)**")
            cm_df = pd.DataFrame(
                st.session_state.analysis_results['cm'], 
                index=['Positive', 'Neutral', 'Negative'],
                columns=['Positive', 'Neutral', 'Negative']
            )
            st.dataframe(cm_df)

            # Visualizations
            st.subheader("Visualizations")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Emotion Distribution Pie Chart**")
                st.pyplot(st.session_state.visualizations['pie_fig'])
                st.download_button(
                    label="Download Pie Chart",
                    data=fig_to_bytes(st.session_state.visualizations['pie_fig']),
                    file_name="pie_chart.png",
                    mime="image/png"
                )
            
            with col2:
                st.write("**Sentiment Distribution Bar Graph**")
                st.pyplot(st.session_state.visualizations['bar_fig'])
                st.download_button(
                    label="Download Bar Graph",
                    data=fig_to_bytes(st.session_state.visualizations['bar_fig']),
                    file_name="bar_graph.png",
                    mime="image/png"
                )

            st.write("**Confusion Matrix**")
            st.pyplot(st.session_state.visualizations['cm_fig'])
            st.download_button(
                label="Download Confusion Matrix",
                data=fig_to_bytes(st.session_state.visualizations['cm_fig']),
                file_name="confusion_matrix.png",
                mime="image/png"
            )

            st.write("**ROC Curve**")
            st.pyplot(st.session_state.visualizations['roc_fig'])
            st.download_button(
                label="Download ROC Curve",
                data=fig_to_bytes(st.session_state.visualizations['roc_fig']),
                file_name="roc_curve.png",
                mime="image/png"
            )

            st.write("**Correlation Matrix of Sentiment Scores**")
            if st.session_state.visualizations['corr_fig']:
                st.pyplot(st.session_state.visualizations['corr_fig'])
                st.download_button(
                    label="Download Correlation Matrix",
                    data=fig_to_bytes(st.session_state.visualizations['corr_fig']),
                    file_name="correlation_matrix.png",
                    mime="image/png"
                )

        elif analyze_button and uploaded_file is None:
            st.error("Please upload a CSV file.")

if __name__ == "__main__":
    main()
