"""Visualize feature importance"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import competition as cc


if __name__ == "__main__":
    feature_importance_df = pd.read_csv(cc.IMPORTANCE_PATH / "{}.csv".format(sys.argv[1]))
    plt.figure(figsize=(14, 26))
    sns.barplot(x="median", y="feature", data=feature_importance_df)
    plt.title("Feature Importance of {}".format(sys.argv[1]))
    plt.tight_layout()
    plt.savefig(str(cc.IMPORTANCE_PATH / "{}.png".format(sys.argv[1])))
