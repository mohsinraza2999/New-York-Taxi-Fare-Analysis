import seaborn as sns
import matplotlib.pyplot as plt
import os

def save_all_plots(final_df, y_test, y_pred, residuals, output_dir='outputs/figures/'):
    os.makedirs(output_dir, exist_ok=True)

    # Residual scatter
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual Fare")
    plt.ylabel("Predicted Fare")
    plt.title("Actual vs Predicted")
    plt.savefig(f"{output_dir}/actual_vs_predicted.png")
    plt.clf()

    # Residual histogram
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residuals Distribution")
    plt.savefig(f"{output_dir}/residuals_hist.png")
    plt.clf()

    # Residuals vs predictions
    sns.scatterplot(x=residuals, y=y_pred)
    plt.xlabel("Residuals")
    plt.ylabel("Predicted")
    plt.title("Residuals vs Predicted")
    plt.savefig(f"{output_dir}/residuals_vs_predicted.png")
    plt.clf()


    # Pairplot
    sns.pairplot(final_df[['fare_amount', 'trip_distance', 'mean_distance']])
    plt.savefig(f"{output_dir}/pairplot.png")
    plt.clf()


    # Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    sns.boxplot(ax=axes[0], x=final_df['trip_distance'])
    sns.boxplot(ax=axes[1], x=final_df['fare_amount'])
    sns.boxplot(ax=axes[2], x=final_df['duration'])
    fig.suptitle("Boxplots for Outliers")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplots_outliers.png")
    plt.clf()

    # Correlation matrix
    """  plt.figure(figsize=(12, 10))
    sns.heatmap(final_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.clf()"""

   

   