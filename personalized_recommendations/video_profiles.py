import pandas as pd

# Paths
file_paths = {
    'metrics_by_country': 'C:\\Users\\prana\\Desktop\\Capstone\\Datasets\\Youtube_Video_Dataset\\Aggregated_Metrics_By_Country_And_Subscriber_Status.csv',
    'metrics_by_video': 'C:\\Users\\prana\\Desktop\\Capstone\\Datasets\\Youtube_Video_Dataset\\Aggregated_Metrics_By_Video.csv',
    'comments': 'C:\\Users\\prana\\Desktop\\Capstone\\Datasets\\Youtube_Video_Dataset\\All_Comments_Final.csv',
    'performance_over_time': 'C:\\Users\\prana\\Desktop\\Capstone\\Datasets\\Youtube_Video_Dataset\\Video_Performance_Over_Time.csv'
}

# Load data
data = {name: pd.read_csv(path, encoding='ISO-8859-1') for name, path in file_paths.items()}

# Calculating net likes gained (likes - dislikes)
data['metrics_by_video']['Net Likes Gained'] = data['metrics_by_video']['Likes'] - data['metrics_by_video']['Dislikes']

# Calculating the net subscribers gained
data['metrics_by_video']['Net Subscribers Gained'] = data['metrics_by_video']['Subscribers gained'] - data['metrics_by_video']['Subscribers lost']

# Group and find top country per video based on views
video_popularity_by_country = data['metrics_by_country'].groupby(['External Video ID', 'Country Code'])['Views'].sum().reset_index()
top_country_per_video = video_popularity_by_country.loc[video_popularity_by_country.groupby('External Video ID')['Views'].idxmax()]

# Calculate average watch time and view percentage
average_watch_time = data['performance_over_time'].groupby('External Video ID')['Average Watch Time'].mean().reset_index()
video_length = data['performance_over_time'][['External Video ID', 'Video Length']].drop_duplicates()
merged_data = pd.merge(average_watch_time, video_length, on='External Video ID', how='inner')
merged_data['Average View Percentage'] = (merged_data['Average Watch Time'] / merged_data['Video Length']) * 100

# Prepare initial performance metrics, adding 'Views' directly without replacing any column
performance_metrics_initial = data['metrics_by_video'][['Video', 'Video title', 'Net Likes Gained', 'Net Subscribers Gained', 'Views']].copy()

# Merge with average view percentage and top country per video
performance_metrics_initial = pd.merge(performance_metrics_initial, merged_data[['External Video ID', 'Average View Percentage']], left_on='Video', right_on='External Video ID', how='left').drop('External Video ID', axis=1)
performance_metrics_initial = pd.merge(performance_metrics_initial, top_country_per_video[['External Video ID', 'Country Code']], left_on='Video', right_on='External Video ID', how='left').rename(columns={'Country Code': 'CountryCode'}).drop('External Video ID', axis=1)

# Calculate Popularity Factor
performance_metrics_initial['Popularity Factor'] = (
    performance_metrics_initial['Net Likes Gained'] +
    performance_metrics_initial['Net Subscribers Gained'] +
    performance_metrics_initial['Views'] +
    performance_metrics_initial['Average View Percentage']
) / 4

# Save to CSV
performance_metrics_initial.to_csv('C:\\Users\\prana\\Desktop\\Capstone\\performance_metrics_updated.csv', index=False)
print(performance_metrics_initial.head())
