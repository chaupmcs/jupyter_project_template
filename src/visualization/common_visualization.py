## last update: 2019-05-29

import math
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.offline as py
import seaborn as sns
import shap
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
# %matplotlib inline
# py.init_notebook_mode(connected=True)

def oneparam_bar_pyplot(data):
    data_label = [go.Bar(
        x=data.value_counts().index,
        y=data.value_counts().values,
        text='Distribution of target variable'
    )]

    layout = go.Layout(
        title='Target variable distribution'
    )

    fig = go.Figure(data=data_label, layout=layout)

    py.iplot(fig, filename='basic-bar')


def oneparam_bar_pyplot_df(data):
    data_label = [go.Bar(
        x=data.index.values,
        y=data.values.reshape(-1, ),
        text='Distribution of target variable'
    )]

    layout = go.Layout(
        title='Target variable distribution'
    )

    fig = go.Figure(data=data_label, layout=layout)

    py.iplot(fig, filename='basic-bar')


def one_param_pie_pyplot(data, hole=0, height=500, width=1200, showlegend=True):
    fig = {
        "data": [
            {
                "values": data.value_counts().values,
                "labels": data.value_counts().index,
                "type": "pie",
                "hole": hole
            }],
        "layout": {"height": height, "width": width, "showlegend": showlegend}
    }
    py.iplot(fig, filename='donut')
    print(data.value_counts())


def one_param_pie_pyplot_df(data, hole=0, height=500, width=1200, showlegend=True):
    """
        ex: oneparam_bar_pyplot_df(df_count.set_index("category"))
    """
    fig = {
        "data": [
            {
                "values": data.values.reshape(-1, ),
                "labels": data.index,
                "type": "pie",
                "hole": hole
            },
        ],
        "layout": {"height": height, "width": width, "showlegend": showlegend}
    }
    py.iplot(fig, filename='donut')


def oneparam_distplot(data, name, bin_size=1):
    print(data[name].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]))
    print("Skewness: %f" % data[name].skew())
    print("Kurtosis: %f" % data[name].kurt())
    fig = ff.create_distplot([data[name].dropna().values], [name], bin_size=bin_size)
    py.iplot(fig, filename='Basic Distplot')


#######################
## 2D:
def twoparam_violinplot(x, y, n):
    for i in range(0, x.shape[1], n):
        df = pd.concat([y, x.iloc[:, i:i + 2]], axis=1)
        x_melt = pd.melt(df, id_vars="label",
                         var_name="features",
                         value_name='value')
        plt.figure(figsize=(20, 20))
        sns.violinplot(x="features", y="value", hue="label", data=x_melt, inner="quart")
        plt.xticks(rotation=90)


def twoparam_swarmplot(x, y, n):
    for i in range(0, x.shape[1], n):
        df = pd.concat([y, x.iloc[:, i:i + 2]], axis=1)
        x_melt = pd.melt(df, id_vars="label",
                         var_name="features",
                         value_name='value')
        plt.figure(figsize=(20, 20))
        sns.swarmplot(x="features", y="value", hue="label", data=x_melt)
        plt.xticks(rotation=90)


def two_param_bar_pyplot(x, y):
    data_label = [go.Bar(
        x=x,
        y=y,
        text='Distribution of target variable'
    )]

    layout = go.Layout(
        title='Target variable distribution'
    )

    fig = go.Figure(data=data_label, layout=layout)

    py.iplot(fig, filename='basic-bar')


def group_bar_pyplot(x, y_1, y_2):
    trace1 = go.Bar(
        x=x,
        y=y_1,
        name=y_1.name
    )
    trace2 = go.Bar(
        x=x,
        y=y_2,
        name=y_2.name
    )
    data = [trace1, trace2]
    layout = go.Layout(
        barmode='group'
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='grouped-bar')


def two_param_pie_pyplot(data, group_name, field_name, hole=0, height=500, width=1200, showlegend=True,
                         textinfo='percent'):
    res = []
    group_list = data[group_name].value_counts().index
    size = math.ceil(len(group_list) / 2)
    for idx, name in enumerate(group_list):
        temp = data[data[group_name] == name][field_name].value_counts()
        res.append({
            "values": temp.values,
            "labels": temp.index,
            "name": name,
            'textinfo': textinfo,
            "domain": {"x": [0 + idx % 2 / 2, 0.5 + idx % 2 / 2],
                       "y": [int(idx / 2) / size, (int(idx / 2) + 1) / size]},
            "hole": hole,
            "type": "pie"})
    fig = {
        "data": res,
        "layout": {"height": height, "width": width, "showlegend": showlegend}
    }
    py.iplot(fig, filename='donut')


def two_param_dist_plot(data, group_name, field_name):
    fig = ff.create_distplot(
        [data[data[group_name] == i][field_name].dropna() for i in data[group_name].value_counts().index],
        data[group_name].value_counts().index)
    py.iplot(fig)


def pair_plot(df, title="Attributes Distribution Pairplot"):
    pp = sns.pairplot(df, size=1.8, aspect=1.8,
                      plot_kws=dict(edgecolor="k", linewidth=0.5),
                      diag_kind="kde", diag_kws=dict(shade=True))

    fig = pp.fig
    fig.subplots_adjust(top=0.93, wspace=0.3)
    t = fig.suptitle(title, fontsize=14)


def plot_confusion_matrix_ver2(df, label_col, pred_col, map_label_to_name, fig_size=(10, 7)):
    """
    df: data frame
    label_col: label column name
    pred_col: predict column name
    map_label_to_name: map numerical labels to string names. Let it blank if no need to set.
                    example: map_label_to_name = {2: "ORANGE", 1:"DURIAN", 0:"APPLE"}
    fig_size: the size of confusion matrix, default: (10,7)
    """

    df_confusion = pd.crosstab(df[label_col], df[pred_col], rownames=['Actual'], colnames=['Predicted'], margins=True)

    if map_label_to_name is None:
        cols = np.unique(df[label_col])
    else:
        cols = [map_label_to_name[c] for c in np.unique(df[label_col])]

    plt.figure(figsize=fig_size)
    sns.set(font_scale=1.0)  # for label size
    f = sns.heatmap(df_confusion, annot=True, annot_kws={"size": 15}, fmt='g', cmap="YlGnBu", xticklabels=cols,
                    yticklabels=cols)  # font size
    return f


def plot_cor_matrix(dataframe, figsize=(10, 8)):
    f, ax = plt.subplots(figsize = figsize)
    corr = dataframe.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot=True)
    
    
def plot_feature_importance_lgb(score, feature_names, num_features_plot = 50, figsize = (15,10), is_returning_df = False):
    lgbm_feature_importance = pd.DataFrame(score, index=feature_names, columns=["feature importance"])
    lgbm_feature_importance["feature importance"] = lgbm_feature_importance["feature importance"] / lgbm_feature_importance["feature importance"].max() #normalize
    top_n = lgbm_feature_importance.nlargest(num_features_plot, columns="feature importance").sort_values(by="feature importance", ascending=True)
    feature_importance_lightgbm = top_n.plot(kind='barh', figsize = figsize)
    if is_returning_df:
        return lgbm_feature_importance.sort_values("feature importance", ascending=False) #return df
    else:
        return feature_importance_lightgbm  ## return the picture

    
def plot_feature_importance_shap(res, max_display = 30):
    '''
    ##### shap for the 5th fold
    X_train = res['X_train']
    regress_model = res["model_list"][-1]
    feature_cols = res["feature_cols"]

    explainer = shap.TreeExplainer(regress_model)
    shap_values = explainer.shap_values(X_train)
    shap.initjs()

    shap.summary_plot(shap_values, X_train, feature_names=feature_cols, max_display = max_display)
    shap.summary_plot(shap_values, X_train, feature_names=feature_cols, max_display = max_display,  plot_type="bar")
    
    ###  multi-class
    #shap.summary_plot(shap_values[1], X_train, feature_names=feature_cols)# impact of each feature on a class in the last fold (in case of multi-class) 
    
    ### dependent plot
    #shap.dependence_plot("phonebook_mean_num_friends_of_friends", shap_values[0], X_train) 
    #shap.dependence_plot("phonebook_mean_num_friends_of_friends", shap_values, X_train)  
    #shap.dependence_plot("count_globaly", shap_values[0], X_train, feature_names=feature_cols)
    '''
    X_train = res['X_train']
    regress_model = res["model_list"][-1]
    feature_cols = res["feature_cols"]

    explainer = shap.TreeExplainer(regress_model)
    shap_values = explainer.shap_values(X_train)
    shap.initjs()

    shap.summary_plot(shap_values, X_train, feature_names=feature_cols, max_display = max_display)
    shap.summary_plot(shap_values, X_train, feature_names=feature_cols, max_display = max_display,  plot_type="bar")
    ###  multi-class
    #shap.summary_plot(shap_values[1], X_train, feature_names=feature_cols)# impact of each feature on a class in the last fold (in case of multi-class) 
    
    ### dependent plot
    #shap.dependence_plot("phonebook_mean_num_friends_of_friends", shap_values[0], X_train) 
    #shap.dependence_plot("phonebook_mean_num_friends_of_friends", shap_values, X_train)  
    #shap.dependence_plot("count_globaly", shap_values[0], X_train, feature_names=feature_cols)
    
def plot_roc(df, label_col, pred_col, filename_if_savetofile=None, fig_size = (20,10)):
    
    fpr, tpr, thresholds = roc_curve(df[label_col].values, df[pred_col].values)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=fig_size)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if filename_if_savetofile:
        plt.savefig(filename_if_savetofile, bbox_inches='tight')
        print("wrote the picture to {}".format(filename_if_savetofile))
    plt.show();

    
    
    
################## confusion matrix -  Thanh
"""
Modify from https://github.com/wcipriano/pretty-print-confusion-matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def _add_margins(df, margins_name="ALL"):
    """Add row and column margins (subtotals)."""
    if isinstance(margins_name, str):
        margins_name = [margins_name] * 2
    elif isinstance(margins_name, list):
        length = len(margins_name)
        if length == 1:
            margins_name = [margins_name] * 2
        elif length < 1 or length > 2:
            raise ValueError("Length of margins_name is {}. "
                             "Expected length is 1 or 2.".format(length))
    else:
        raise ValueError("margins_name argument must be a string or a list.")

    # Check for name conflicts
    row_name, col_name = margins_name
    if row_name in df.index.values:
        raise ValueError('Index name "{}" already existed.'.format(row_name))
    if col_name in df.columns.values:
        raise ValueError('Column name "{}" already existed.'.format(col_name))
    
    # Compute subtotal row and column
    sum_rows = np.sum(df, axis=0)
    sum_cols = np.sum(df, axis=1)
    grand_total = np.sum(sum_rows)

    result = df.copy()
    result[col_name] = sum_cols
    sum_rows = np.append(sum_rows, grand_total)
    result.loc[row_name] = sum_rows
    return result


def _config_cell_properties(data, x, y, position, text, fontsize, fmt,
                            facecolors, show_null_values=False):
    text_add = []
    text_del = []
    
    cell_value = data[y][x]
    total = data[-1][-1]
    cell_percentage = (float(cell_value) / total) * 100
    
    current_column = data[:, y]
    column_length = len(current_column)
    
    # Config for last row and/or last column
    if (x == column_length - 1) or (y == column_length - 1):
        if cell_value != 0:
            if (x == column_length - 1) and (y == column_length - 1):
                total_right = np.sum(data.diagonal()[:-1])
            elif x == column_length - 1:
                total_right = data[y][y]
            elif y == column_length - 1:
                total_right = data[x][x]
            right_percentage = (float(total_right) / cell_value) * 100
            wrong_percentage = 100 - right_percentage
        else:
            right_percentage = wrong_percentage = 0

        # Delete old text
        text_del.append(text)
        
        # Add new text
        list_text = [
            format(cell_value, ",d"),
            format(right_percentage, fmt) + "%",
            format(wrong_percentage, fmt) + "%"
        ]

        text_properties = dict(
            color="w", ha="center", va="center", gid="sum",
            fontproperties=fm.FontProperties(weight="bold", size=fontsize)
        )
        list_text_properties = [text_properties.copy() for i in range(3)]
        list_text_properties[1]["color"] = "xkcd:vibrant green"
        list_text_properties[2]["color"] = "red"

        text_x, text_y = text.get_position()
        list_positions = [
            (text_x, text_y - 0.3),
            (text_x, text_y),
            (text_x, text_y + 0.3)
        ]
        
        for i, text in enumerate(list_text):
            new_text = dict(
                x=list_positions[i][0], y=list_positions[i][1],
                text=list_text[i], kw=list_text_properties[i]
            )
            text_add.append(new_text)
        
        # Set background color for cells at the margin
        background_color = [0.27, 0.30, 0.27, 0.85]  # RGBA
        if (x == column_length - 1) and (y == column_length - 1):
            background_color = [0.17, 0.20, 0.17, 0.85]
        facecolors[position] = background_color
    
    else:
        if cell_percentage > 0:
            new_text = format(cell_value, ",d") + \
                       "\n" + format(cell_percentage, fmt) + "%"
        elif show_null_values:
            new_text = "0\n" + format(0, fmt)
        else:
            new_text = ""
        text.set_text(new_text)
        
        # Main diagonal
        if (x == y):
            # Set color of the text in the diagonal cells
            text.set_color("w")
            # Set background color for the diagonal cells
            facecolors[position] = [0.35, 0.8, 0.55, 1.0]
        else:
            text.set_color("r")    
        
    return text_add, text_del

    
def plot_precomputed_confusion_matrix(
    cm, title="Confusion Matrix", figsize=(10, 10),
    axis_labels=("Predicted", "Actual"), predict_axis="x",
    fontsize=10, fmt=",.2f", cmap="Oranges", cbar=False,
    linewidths=0.5, show_null_values=False, ax=None):
    """
    Plot confusion matrix from precomputed data.
    
    Parameters
    ----------
    cm : ndarray, Pandas DataFrame
        2D dataset that can be coerced into an ndarray. If a Pandas DataFrame is
        is provided, the index/column information will be used to label the
        columns and rows.

    title : str, optional, default: "Confusion Matrix"
        Text to use for the title.

    figsize : (float, float), optional, default: (10, 10)
        Figure width and height in inches.
    
    axis_labels : (str, str), optional, default: ("Predicted", "Actual")
        A tuple of explicit labels for the x-axis and y-axis.

    predict_axis : {"x", "X", "y", "Y"}, optional, default: "x"
        Whether to use x-axis or y-axis as predict axis.

    fontsize : {size in points, "xx-small", "x-small", "small", "medium", \
"large", "x-large", "xx-large"}, optional, default: 10

    fmt : str, optional, default: ",.2f"
        String formatting code to use when adding percentage annotations.

    cmap : matplotlib colormap name/object, optional, default: "Oranges"
        The mapping from data values to color space.

    cbar : boolean, optional, default: False
        Whether to draw a colorbar.

    linewidths : float, optional, default: 0.5
        Width of the lines that will divide each cell.

    show_null_values : boolean, optional, default: False
        Whether to annotate cell with null values.

    ax : matplotlib Axes, optional, default: None
        Axes in which to draw the plot, otherwise create a new Axes.
    
    Returns
    -------
    ax : matplotlib Axes
        Axes object with the confusion matrix.
    """
    if predict_axis in ("x", "X"):
        x_label, y_label = axis_labels
        margins_name = ["PRECISION", "RECALL"]
    elif predict_axis in ("y", "Y"):
        cm = cm.T
        y_label, x_label = axis_labels
        margins_name = ["RECALL", "PRECISION"]
    else:
        raise ValueError('predict_axis argument must be "x" or "y"')
    
    if ax is None:
        fig = plt.figure(title, figsize=figsize)
        ax = fig.gca()  # get current axis
        ax.cla()        # clear existing plot
    
    # Insert summary row and column (subtotals)
    cm = _add_margins(cm, margins_name=margins_name)
    
    # Plot confusion matrix
    ax = sns.heatmap(cm, annot=True, annot_kws={"size": fontsize}, fmt=fmt,
                     square=True, cbar=cbar, cmap=cmap,
                     linecolor="w", linewidths=linewidths, ax=ax)
    
    # Turn of all the ticks
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    
    # Set labels rotation
    ax.tick_params(axis='x', rotation=90, labelsize=fontsize+2)
    ax.tick_params(axis='y', rotation=0, labelsize=fontsize+2)
        
    # Face color list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()
    
    # Text annotation
    cm = cm.to_numpy()
    text_add = []
    text_del = []
    position = -1  # from left to right, bottom to top
    
    for text in ax.collections[0].axes.texts:
        pos = text.get_position() - np.array([0.5, 0.5])
        x, y = pos.astype(int)
        position += 1

        # This will modify `text` and `facecolors` inplace
        text_to_add, text_to_delete = \
            _config_cell_properties(cm, x=x, y=y, position=position,
                                    text=text,fontsize=fontsize, fmt=fmt,
                                    facecolors=facecolors,
                                    show_null_values=show_null_values)
        text_add.extend(text_to_add)
        text_del.extend(text_to_delete)

    # Remove old text and add new text
    for text in text_del:
        text.remove()
    for text in text_add:
        ax.text(text["x"], text["y"], text["text"], **text["kw"])
    
    # Set title and legends
    ax.set_title(title, fontsize=fontsize+7, pad=20)
    ax.set_xlabel(x_label, fontsize=fontsize+5)
    ax.set_ylabel(y_label, fontsize=fontsize+5)
    return ax


def plot_confusion_matrix_ver3(y_true, y_pred, **kwargs):
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.
        
    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.

    title : str, optional, default: "Confusion Matrix"
        Text to use for the title.

    figsize : (float, float), optional, default: (10, 10)
        Figure width and height in inches.
    
    axis_labels : (str, str), optional, default: ("Predicted", "Actual")
        A tuple of explicit labels for the x-axis and y-axis.

    predict_axis : {"x", "X", "y", "Y"}, optional, default: "x"
        Whether to use x-axis or y-axis as predict axis.

    fontsize : {size in points, "xx-small", "x-small", "small", "medium", \
"large", "x-large", "xx-large"}, optional, default: 10

    fmt : str, optional, default: ",.2f"
        String formatting code to use when adding percentage annotations.

    cmap : matplotlib colormap name/object, optional, default: "Oranges"
        The mapping from data values to color space.

    cbar : boolean, optional, default: False
        Whether to draw a colorbar.

    linewidths : float, optional, default: 0.5
        Width of the lines that will divide each cell.

    show_null_values : boolean, optional, default: False
        Whether to annotate cell with null values.

    ax : matplotlib Axes, optional, default: None
        Axes in which to draw the plot, otherwise create a new Axes.
    
    Returns
    -------
    ax : matplotlib Axes
        Axes object with the confusion matrix.
    """
    classes = unique_labels(y_true, y_pred)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred),
                      columns=classes, index=classes)
    return plot_precomputed_confusion_matrix(cm, **kwargs)


def plot_fold_result(score_list,  final_score, num_folds, figsize=(10,6), pic_name = "", yaxis_range = (0,100), ylabel = "AUC Score", set_name = ""):
    """
        plot_fold_result(fold_auc_score_list, auc_score, 5, figsize=(10,6), pic_name = "", yaxis_range = (0,100), ylabel = "AUC Score", set_name = "Train")
    
    """
    
    plt.figure(figsize = figsize)
    ax = sns.barplot(x=list(range(1, 1+num_folds)), y=score_list)

    for p in ax.patches:
                 ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=16, color='blue', xytext=(0, 20),
                     textcoords='offset points')
    ax.set_ylim(*yaxis_range) #To make space for the annotations
    plt.ylabel(ylabel)
    plt.xlabel('Model')
    plt.title('AUC of 5 models on {} Set (std = {}; {} = {})'.format(set_name,
        np.round(np.std(score_list), 2), ylabel, np.round(final_score, 2)))
    if pic_name != "":
        plt.savefig(pic_name, bbox_inches='tight')