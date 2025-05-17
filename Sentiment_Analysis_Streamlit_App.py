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

# Read social media posts from CSV file and generate true labels
def read_posts(file, text_column, pos_threshold, neg_threshold):
    try:
        df = pd.read_csv(file)

        # Check if text column exists
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset.")

        # Display the first five rows
        st.subheader("First Five Rows of Loaded CSV")
        st.dataframe(df.head())

        # Remove duplicate records
        df = df.drop_duplicates(subset=text_column)

        # Drop missing (null) values
        df = df[[text_column]].dropna()

        # Convert to string format and generate true labels using VADER
        posts = df[text_column].astype(str).tolist()
        true_labels = []
        for post in posts:
            score = analyzer.polarity_scores(post)['compound']
            true_labels.append(decode_emotion(score, pos_threshold, neg_threshold))

        st.success(f"Total posts loaded: {len(posts)}")
        return posts, true_labels, df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None, None, None

# Decode sentiment score to emotion
def decode_emotion(score, pos_threshold, neg_threshold):
    if score >= pos_threshold:
        return "Positive"
    elif score <= neg_threshold:
        return "Negative"
    else:
        return "Neutral"

# Analyze sentiments for CSV posts
def analyze_posts(posts, pos_threshold, neg_threshold):
    predicted_labels = []
    compound_scores = []
    polarity_scores = {'compound': [], 'pos': [], 'neu': [], 'neg': []}
    results_text = ""
    analysis_data = []
    for post in posts:
        scores = analyzer.polarity_scores(post)
        score = scores['compound']
        emotion = decode_emotion(score, pos_threshold, neg_threshold)
        predicted_labels.append(emotion)
        compound_scores.append(score)
        polarity_scores['compound'].append(scores['compound'])
        polarity_scores['pos'].append(scores['pos'])
        polarity_scores['neu'].append(scores['neu'])
        polarity_scores['neg'].append(scores['neg'])
        results_text += f"{post[:50]}...\n-> {emotion} (Score: {score:.3f})\n\n"
        analysis_data.append({'Post': post, 'Sentiment': emotion, 'Compound Score': score})
    return predicted_labels, compound_scores, polarity_scores, results_text, analysis_data

# Analyze single text input
def analyze_single_text(text, pos_threshold, neg_threshold):
    if not text.strip():
        return "Error: Please enter some text."
    scores = analyzer.polarity_scores(text)
    score = scores['compound']
    emotion = decode_emotion(score, pos_threshold, neg_threshold)
    return f"{text[:50]}...\n-> {emotion} (Score: {score:.3f})"

# Compute evaluation metrics for CSV analysis
def compute_metrics(true_labels, predicted_labels, compound_scores):
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    label_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    true_numeric = [label_mapping[label] for label in true_labels]
    predicted_numeric = [label_mapping[label] for label in predicted_labels]
    rmse = np.sqrt(mean_squared_error(true_numeric, predicted_numeric))
    cm = confusion_matrix(true_labels, predicted_labels, labels=['Positive', 'Neutral', 'Negative'])
    binary_true = [1 if label == 'Positive' else 0 for label in true_labels]
    fpr, tpr, _ = roc_curve(binary_true, compound_scores)
    roc_auc = auc(fpr, tpr)
    return accuracy, f1, rmse, cm, fpr, tpr, roc_auc

# Display pie chart
def create_pie_chart(results):
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = results.keys()
    sizes = results.values()
    colors = ['green', 'gray', 'red']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.set_title("Emotion Distribution in Social Media Posts")
    ax.axis('equal')
    return fig

# Display bar graph
def create_bar_graph(results):
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list(results.keys())
    counts = list(results.values())
    colors = ['green', 'gray', 'red']
    ax.bar(labels, counts, color=colors)
    ax.set_title('Sentiment Distribution in Social Media Posts')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Posts')
    return fig

# Display confusion matrix
def create_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Positive', 'Neutral', 'Negative'],
                yticklabels=['Positive', 'Neutral', 'Negative'], ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True Label (Auto-Generated)')
    ax.set_xlabel('Predicted Label')
    return fig

# Display ROC curve
def create_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (Positive vs Non-Positive)')
    ax.legend(loc="lower right")
    return fig

# Display correlation matrix
def create_correlation_matrix(polarity_scores):
    if not polarity_scores['compound']:
        st.error("No data available for correlation matrix.")
        return None
    df_scores = pd.DataFrame(polarity_scores, columns=['compound', 'pos', 'neu', 'neg'])
    corr_matrix = df_scores.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title('Correlation Matrix of Sentiment Scores')
    return fig

# Convert figure to bytes for download
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()

# Main app
def main():
    st.title("📊 Sentiment Analysis App")
    st.markdown("""
    Use the tabs below to analyze a single text post or a CSV file containing social media posts.
    Adjust sentiment thresholds as needed. The app uses VADER to analyze sentiments and display results.
    Download results and visualizations as needed.
    """)

    # Initialize session state for CSV analysis
    if 'csv_analyzed' not in st.session_state:
        st.session_state.csv_analyzed = False
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = None

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
        if (analyze_button and uploaded_file is not None) or st.session_state.csv_analyzed:
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

                        # Store data in session state
                        st.session_state.csv_analyzed = True
                        st.session_state.analysis_data = {
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
                        st.session_state.visualizations = {
                            'pie_fig': create_pie_chart(results),
                            'bar_fig': create_bar_graph(results),
                            'cm_fig': create_confusion_matrix(cm),
                            'roc_fig': create_roc_curve(fpr, tpr, roc_auc),
                            'corr_fig': create_correlation_matrix(polarity_scores)
                        }

            # Display results if analysis data exists
            if st.session_state.analysis_data is not None:
                st.subheader("CSV Analysis Results")
                
                # Sentiment analysis output
                with st.expander("Sentiment Analysis Output", expanded=False):
                    st.text_area("Post Sentiments", st.session_state.analysis_data['results_text'], height=300)
                # Download sentiment analysis results
                analysis_df = pd.DataFrame(st.session_state.analysis_data['analysis_data'])
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
                    st.metric("Accuracy", f"{st.session_state.analysis_data['accuracy']:.3f}")
                    st.metric("F1 Score (Weighted)", f"{st.session_state.analysis_data['f1']:.3f}")
                with col2:
                    st.metric("RMSE", f"{st.session_state.analysis_data['rmse']:.3f}")
                    st.metric("ROC AUC", f"{st.session_state.analysis_data['roc_auc']:.3f}")
                
                # Download metrics report
                metrics_report = f"""Sentiment Analysis Metrics Report
Accuracy: {st.session_state.analysis_data['accuracy']:.3f}
F1 Score (Weighted): {st.session_state.analysis_data['f1']:.3f}
RMSE: {st.session_state.analysis_data['rmse']:.3f}
ROC AUC: {st.session_state.analysis_data['roc_auc']:.3f}
"""
                st.download_button(
                    label="Download Metrics Report",
                    data=metrics_report,
                    file_name="metrics_report.txt",
                    mime="text/plain"
                )

                st.write("**Confusion Matrix (Rows: True, Columns: Predicted)**")
                cm_df = pd.DataFrame(
                    st.session_state.analysis_data['cm'], 
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
