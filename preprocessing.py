import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# 🔹 Load Dataset
df = pd.read_csv("balanced_cleaned_dataset.csv")

def generate_graphs(df):
    os.makedirs("static/graphs", exist_ok=True)

    # GPA banding
    bands = {
        "Poor": (0, 4.99),
        "Average": (5.0, 6.49),
        "Good": (6.5, 7.99),
        "Excellent": (8.0, 10.0)
    }

    def assign_band(gpa):
        for label, (low, high) in bands.items():
            if low <= gpa <= high:
                return label
        return "Unknown"

    df["GPA_Band"] = df["GPA"].apply(assign_band)

    # Simulated normalized user input
    example = {
        "HSC_Score": 0.82,
        "SSC_Score": 0.75,
        "Attendance": 0.9,
        "English_Proficiency": 0.8,
        "Daily_Study_Hours": 0.7
    }
    example["Study_Efficiency"] = example["Daily_Study_Hours"] * example["Attendance"]
    example["Exam_Proficiency"] = (example["HSC_Score"] + example["SSC_Score"]) / 2
    example["English_Impact"] = example["English_Proficiency"] * 0.78
    example_df = pd.DataFrame([example])

    # Graph definitions
    graph_defs = [
        ("HSC_Score", "HSC Score by Performance Category", "hsc_score_box.png", "boxplot"),
        ("SSC_Score", "SSC Distribution by Performance Category", "ssc_histogram.png", "hist"),
        ("Study_Efficiency", "Study Efficiency vs GPA", "study_scatter.png", "scatter"),
        ("English_Impact", "English Impact per Category", "english_violin.png", "violin"),
        ("Attendance", "Attendance vs GPA Category", "attendance_strip.png", "strip"),
        ("Exam_Proficiency", "Exam Proficiency Density by GPA Category", "exam_kde.png", "kde"),
        ("English_Proficiency", "English Proficiency by GPA Category", "english_swarm.png", "swarm"),
        ("Daily_Study_Hours", "Daily Study Hours per GPA Category", "study_bar.png", "bar"),
        ("GPA", "GPA Distribution", "gpa_distplot.png", "dist"),
        ("SSC_Score", "SSC Score ECDF by GPA Band", "ssc_ecdf.png", "ecdf")
    ]

    for feature, title, filename, kind in graph_defs:
        plt.figure(figsize=(8, 5))

        if kind == "boxplot":
            sns.boxplot(x="GPA_Band", y=feature, data=df, order=["Poor", "Average", "Good", "Excellent"], palette="Set2")
        elif kind == "hist":
            for band in df["GPA_Band"].unique():
                subset = df[df["GPA_Band"] == band]
                sns.histplot(subset[feature], kde=True, label=band, element="step", stat="density", fill=False)
            plt.legend(title="GPA Category")
        elif kind == "scatter":
            sns.scatterplot(x=feature, y="GPA", data=df, hue="GPA_Band", palette="coolwarm", alpha=0.6)
        elif kind == "violin":
            sns.violinplot(x="GPA_Band", y=feature, data=df, order=["Poor", "Average", "Good", "Excellent"], palette="muted")
        elif kind == "strip":
            sns.stripplot(x="GPA_Band", y=feature, data=df, order=["Poor", "Average", "Good", "Excellent"], palette="Set3", jitter=True)
        elif kind == "kde":
            sns.kdeplot(data=df, x=feature, hue="GPA_Band", common_norm=False, fill=True, alpha=0.3)
        elif kind == "swarm":
            sns.swarmplot(x="GPA_Band", y=feature, data=df, order=["Poor", "Average", "Good", "Excellent"], palette="husl")
        elif kind == "bar":
            avg_values = df.groupby("GPA_Band")[feature].mean().reindex(["Poor", "Average", "Good", "Excellent"])
            sns.barplot(x=avg_values.index, y=avg_values.values, palette="Blues_d")
        elif kind == "dist":
            sns.histplot(df[feature], kde=True, color="purple")
        elif kind == "ecdf":
            sns.ecdfplot(data=df, x=feature, hue="GPA_Band")

        if feature in example_df.columns:
            user_val = example_df[feature].values[0]
            min_val, max_val = df[feature].min(), df[feature].max()
            user_val = np.clip(user_val, min_val, max_val)

            if kind in ["boxplot", "violin", "strip", "swarm", "bar"]:
                plt.axhline(user_val, color='red', linestyle='--', linewidth=2, clip_on=True)
            elif kind in ["scatter", "hist", "kde", "dist", "ecdf"]:
                plt.axvline(user_val, color='red', linestyle='--', linewidth=2, label='User Input')
                plt.legend()

        plt.title(title)
        plt.xlabel("GPA Band" if kind in ["boxplot", "violin", "strip", "swarm", "bar"] else feature)
        plt.ylabel(feature if kind in ["boxplot", "violin", "strip", "swarm", "bar"] else "GPA")
        plt.tight_layout()
        plt.savefig(f"static/graphs/{filename}")
        plt.close()

# 🧪 Run Graph Generation
generate_graphs(df)

