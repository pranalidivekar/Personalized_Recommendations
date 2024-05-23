import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 16})
def plot_precision_recall_for_users(precisions, recalls, user_ids):
    """
    Plots precision and recall for a list of users.
    
    :param precisions: List of precision values for users.
    :param recalls: List of recall values for users.
    :param user_ids: List of user identifiers.
    """
    # Setting up the figure and the bar width
    n_users = len(user_ids)
    bar_width = 0.35
    index = np.arange(n_users)
    
    # Creating the bars for precision and recall
    plt.figure(figsize=(10, 6))
    bars_precision = plt.bar(index, precisions, bar_width, label='Precision')
    bars_recall = plt.bar(index + bar_width, recalls, bar_width, label='Recall')
    
    # Adding some presentation logic
    plt.xlabel('User')
    plt.ylabel('Scores')
    plt.title('Precision and Recall for Different Users')
    plt.xticks(index + bar_width / 2, user_ids)  # Place x-axis labels in the middle of the group of bars
    plt.legend(ncol=2)
    
    plt.tight_layout()
    plt.show()

def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def plot_f1_scores_for_users(f1_scores, user_ids):
    """
    Plots F1 scores for a list of users.
    
    :param f1_scores: List of F1 scores for users.
    :param user_ids: List of user identifiers.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(user_ids, f1_scores, color='#007ACC', marker='o', linestyle='-', label='F1 Score')

    plt.xlabel('User')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores for Different Users')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    videos_df = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\enhanced_performance_metrics_with_topics.csv')
    features = ['Net Likes Gained', 'Net Subscribers Gained', 'Average View Percentage']
    for feature in features:
        videos_df[feature] = pd.to_numeric(videos_df[feature], errors='coerce')
    # Drop rows with any NaN values in the specified features
    videos_df.dropna(subset=features, inplace=True)
    scaler = MinMaxScaler()
    videos_df[features] = scaler.fit_transform(videos_df[features])
    videos_df['relevancy_factor'] = videos_df['Net Likes Gained'] * 0.5 + videos_df['Net Subscribers Gained'] * 0.3 + videos_df['Views'] * 0.2
    # threshold
    threshold = videos_df['relevancy_factor'].quantile(0.50)
    #print("Threshold=",threshold)
    videos_df['is_relevant'] = videos_df['relevancy_factor'] >= threshold
    # For simplicity, let's assume recommended_videos for a single user
    recommended_videos = ['-3d1NctSv0c', 'Ip50cXvpWY4', '742LQ38OioU', 'Xgg7dIKys9E', '4qZINLzwYyk']

    # Marking recommendations as relevant based on 'is_relevant' flag
    relevant_recommended = videos_df[videos_df['Video'].isin(recommended_videos) & videos_df['is_relevant']]

    # Calculating precision
    precision = len(relevant_recommended) / len(recommended_videos)
    total_relevant = videos_df['is_relevant'].sum()
    recall = len(relevant_recommended) / total_relevant


    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}')

    user_ids = [f'User {i+1}' for i in range(5)]  # Example user IDs
    #precisions = [0.9, 0.85, 0.8, 0.75, 0.7]  # Sample precision values for 10 users
    #recalls = [0.5, 0.55, 0.6, 0.45, 0.5]  # Sample recall values for 10 users
    precisions = [0.20, 0.21, 0.22, 0.23, 0.24,]
    recalls = [0.02, 0.021, 0.022, 0.023, 0.024,]
     # Calculating F1 score for each user
    f1_scores = [calculate_f1_score(p, r) for p, r in zip(precisions, recalls)]
    plot_precision_recall_for_users(precisions, recalls, user_ids)
    plot_f1_scores_for_users(f1_scores, user_ids)
if __name__ == "__main__":
    main()


#run for many users, calculate precisions and recall values, plot for these values