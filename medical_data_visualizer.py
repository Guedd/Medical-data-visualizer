from unittest import main
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("C:/Users/DJEF/Downloads/medical_examination.csv")

# Add 'overweight' column
df['overweight'] = (df.weight / np.square(df.height / 100) > 25).astype(int) 

print(df.columns)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df.cholesterol = (df.cholesterol > 1).astype(int)
df.gluc = (df.gluc > 1).astype(int)
print(df.gluc)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df.melt(['cardio'], df_cat)

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(data=df_cat, kind='count', x='variable', hue='value',col="cardio")

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = (
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025))&
        (df['height'] <= df['height'].quantile(0.095))&
        (df['weight'] >= df['weight'].quantile(0.025))&
        (df['weight'] <= df['weight'].quantile(0.095))
    )

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.trui(corr)



    # Set up the matplotlib figure
    fig, ax = ax = sns.heatmap(
        corr, annot=True, fmt='.1f', linewidths=1, mask=mask, 
        vmax=.8, center=0.09, square=True, cbar_kws={'shrink':0.5})

    # Draw the heatmap with 'sns.heatmap()'



    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
