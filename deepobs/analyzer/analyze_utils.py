import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib2tikz import get_tikz_code

def beautify_lr_sensitivity(fig, ax):
    """Beautify a learning rate sensitivity plot.

    This function adds axis labels and removes spines to create a nicer learning
    rate sensitivity plot.

    Args:
        fig (matplotlib.figure): Handle to the matplotlib figure of the learning
            rate sensitivity plot.
        ax (list): List of lists of matplotlib axis of the learning rate
            sensitivity plots.

    Returns:
        matplotlib.figure: Handle to the beautified matplotlib figure of the
        learning rate sensitivity plot.
        list: List of lists of the beautified matplotlib axis of the learning
        rate sensitivity plots.

    """
    fig.suptitle("Learning rate sensitivity", fontsize=20)
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i][j].get_yaxis().set_visible(False)
            ax[i][j].spines['top'].set_visible(False)
            ax[i][j].spines['right'].set_visible(False)
            #     ax[i][j].spines['bottom'].set_visible(False)
            ax[i][j].spines['left'].set_visible(False)
            if i == 0:
                ax[i][j].get_xaxis().set_visible(False)
            if i == 1:
                ax[i][j].set_xlabel('Learning Rate')
    return fig, ax


def texify_lr_sensitivity(fig, ax):
    """Write a ``.tex`` file with the learning rate sensitivity plot.

    The function will create a file named `tuning_plot.tex` with the latex code
    for the learning rate sensitivity plot.

    Args:
        fig (matplotlib.figure): Handle to the matplotlib figure of the learning
            rate sensitivity plot.
        ax (list): List of lists of matplotlib axis of the learning rate
            sensitivity plots.

    Returns:
        str: String of the latex code for the learning rate sensitivity plot.

    """
    tikz_code = get_tikz_code(
        'tuning_plot_new.tex',
        figureheight='\\figureheight',
        figurewidth='0.33\\figurewidth')

    tikz_code = tikz_code.replace(
        '\\begin{groupplot}[group style={group size=4 by 2}]',
        '\\begin{groupplot}[group style={group size=4 by 2, horizontal sep=0.02\\figurewidth, vertical sep=0.15cm}]'
    )
    tikz_code = r"\pgfplotsset{every axis/.append style={label style={font=\tiny}, tick label style={font=\tiny}, legend style={font=\tiny, line width=1pt}}}" + tikz_code
    tikz_code = tikz_code.replace('minor', '%minor')  # comment minor tick
    tikz_code = tikz_code.replace('x grid',
                                  '%x grid')  # remove grid xmajorticks=false,
    tikz_code = tikz_code.replace('y grid', '%y grid')  # remove grid
    tikz_code = tikz_code.replace('tick align',
                                  '%tick align')  # ugly outside ticks
    tikz_code = tikz_code.replace(
        'nextgroupplot[', 'nextgroupplot[axis x line*=bottom,\nhide y axis,'
    )  # ugly outside ticks
    tikz_code = tikz_code.replace(
        '(current bounding box.south west)!0.98!(current bounding box.north west)',
        '(current bounding box.south west)!1.05!(current bounding box.north west)'
    )  # position title higher
    tikz_code = tikz_code.replace('title={',
                                  'title={\small ')  # shrink title size

    # Write the file out again
    with open('tuning_plot.tex', 'w') as file:
        file.write(tikz_code)

    return tikz_code


def rescale_ax(ax):
    """Rescale an axis to include the most important data.

    Args:
        ax (matplotlib.axis): Handle to a matplotlib axis.

    """
    lines = ax.lines
    y_data = np.array([])
    y_limits = []
    for line in lines:
        if line.get_label() != "convergence_performance":
            y_data = np.append(y_data, line.get_ydata())
        else:
            y_limits.append(line.get_ydata()[0])
    if len(y_data)!=0:
        y_limits.append(np.percentile(y_data, 20))
        y_limits.append(np.percentile(y_data, 80))
        y_limits = [y_limits[0] * 0.9, y_limits[1] * 1.1]
        if y_limits[0] != y_limits[1]:
            ax.set_ylim([max(1e-10, y_limits[0]), y_limits[1]])
        ax.margins(x=0)
    else:
        ax.set_ylim([1.0, 2.0])
    return ax

def beautify_plot_performance(fig, ax, folder_parser, problem_set):
    """Beautify a performance plot.

    This function adds axis labels, sets titles and more to create a nicer
    performance plot.

    Args:
        fig (matplotlib.figure): Handle to the matplotlib figure of the
            performance plot.
        ax (list): List of lists of matplotlib axis of the performance plot.
        folder_parser (Analyzer): An instance of the DeepOBS Analyzer class
            to plot the performance from.
        problem_set (str): Can either be ``small`` or ``large`` to switch
            between which benchmark set is being plotted.

    Returns:
        matplotlib.figure: Handle to the beautified matplotlib figure of the
        performance plot.
        list: List of lists of the beautified matplotlib axis of the performance
        plots.

    """
    fig.subplots_adjust(hspace=0.4)
    if problem_set == "small":
        fig.suptitle("Benchmark Set Small", fontsize=20)
        titles = [
            "P1 Quadratic Deep", "P2 MNIST - VAE", "P3 F-MNIST - CNN",
            "P4 CIFAR-10 - CNN"
        ]
        # clear axis (needed for matplotlib2tikz)
        plt.sca(ax[2][0])
        plt.cla()
        plt.sca(ax[2][1])
        plt.cla()
        plt.sca(ax[3][1])
        plt.cla()
        ax[2][1].axis('off')
        ax[3][1].axis('off')
        ax[1][0].set_xlabel("Epochs")
        ax[1][1].set_xlabel("Epochs")
        ax[2][2].set_ylabel("Test Accuracy")
        ax[3][2].set_ylabel("Train Accuracy")
        ax[1][1].tick_params(
            axis='x', which='major', bottom=False,
            labelbottom=True)  # show x axis
        # Add convergence performance line
#        for idx, tp in enumerate(
#            ["quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"]):
#            if tp in folder_parser.testproblems:
#                metric = folder_parser.testproblems[tp].metric
#                conv_perf = folder_parser.testproblems[tp].conv_perf
#                if metric == "test_losses":
#                    ax_row = 0
#                elif metric == "test_accuracies":
#                    ax_row = 2
#                ax[ax_row][idx].axhline(
#                    conv_perf, color='#AFB3B7', label="convergence_performance")
    elif problem_set == "large":
        fig.suptitle("Benchmark Set Large", fontsize=20)
        ax[1][0].set_xlabel("Epochs")
        ax[3][1].set_xlabel("Epochs")
        ax[2][1].set_ylabel("Test Accuracy")
        ax[3][1].set_ylabel("Train Accuracy")
        titles = [
            "P5 F-MNIST - VAE", "P6 CIFAR 100 - All CNN C",
            "P7 SVHN - Wide ResNet 16-4", "P8 Tolstoi - Char RNN"
        ]
        # Add convergence performance line
#        for idx, tp in enumerate([
#                "fmnist_vae", "cifar100_allcnnc", "svhn_wrn164",
#                "tolstoi_char_rnn"
#        ]):
#            if tp in folder_parser.testproblems:
#                metric = folder_parser.testproblems[tp].metric
#                conv_perf = folder_parser.testproblems[tp].conv_perf
#                if metric == "test_losses":
#                    ax_row = 0
#                elif metric == "test_accuracies":
#                    ax_row = 2
#                ax[ax_row][idx].axhline(
#                    conv_perf, color='#AFB3B7', label="convergence_performance")
    # clear axis (needed for matplotlib2tikz)
    plt.sca(ax[2][0])
    plt.cla()
    plt.sca(ax[3][0])
    plt.cla()
    ax[2][0].axis('off')
    ax[3][0].axis('off')
    ax[3][2].set_xlabel("Epochs")
    ax[3][3].set_xlabel("Epochs")
    ax[0][0].set_ylabel("Test Loss")
    ax[1][0].set_ylabel("Train Loss")
    ax[1][0].tick_params(
        axis='x', which='major', bottom=False, labelbottom=True)  # show x axis
    # automatic rescaling
    for axlist in ax:
        for a in axlist:
            a = rescale_ax(a)
    # Legend
    handles, labels = ax[0][3].get_legend_handles_labels()
    #     labels_tex = [tfobs.plot_utils.texify(l) for l in labels]
    ax[3][0].legend(
        handles,
        labels,
        loc='upper right',
        bbox_to_anchor=(0.2, 1.1, 0.5, 0.5))
    for idx, title in enumerate(titles):
        ax[0, idx].set_title(title)
    return fig, ax


def texify_plot_performance(fig, ax, problem_set):
    """Write a ``.tex`` file with the performance plot.

    The function will create a file named `benchmark_small.tex` or
    `benchmark_large.tex` with the latex code for the performance plot.

    Args:
        fig (matplotlib.figure): Handle to the matplotlib figure of the
            performance plot.
        ax (list): List of lists of matplotlib axis of the performance plot.
        problem_set (str): Can either be ``small`` or ``large`` to switch
            between which benchmark set is being plotted.

    Returns:
        str: String of the latex code for the learning rate sensitivity plot.

    """
    file_name = 'benchmark_' + str(problem_set) + '.tex'
    tikz_code = get_tikz_code(
        fig, figureheight='\\figureheight', figurewidth='\\figurewidth')

    tikz_code = r"\pgfplotsset{every axis/.append style={label style={font=\tiny}, tick label style={font=\tiny}, legend style={font=\tiny, line width=1pt}}}" + tikz_code
    tikz_code = tikz_code.replace('minor', '%minor')  # comment minor tick
    tikz_code = tikz_code.replace('x grid', '%x grid')  # remove grid
    tikz_code = tikz_code.replace('y grid', '%y grid')  # remove grid
    tikz_code = tikz_code.replace('tick align',
                                  '%tick align')  # ugly outside ticks
    tikz_code = tikz_code.replace(
        'nextgroupplot[',
        'nextgroupplot[axis x line*=bottom,\naxis y line*=left,'
    )  # ugly outside ticks
    tikz_code = tikz_code.replace('xlabel={Epochs},\nxmajorticks=false,',
                                  'xlabel={Epochs},\nxmajorticks=true,'
                                  )  # if x label is epoch, show ticks
    tikz_code = tikz_code.replace('ymajorticks=false,',
                                  'ymajorticks=true,')  # show y labels
    tikz_code = tikz_code.replace('\mathdefault',
                                  '')  # remove mathdefault in labels
    tikz_code = tikz_code.replace(
        '\path [draw=white!80.0!black, fill opacity=0]',
        '%\path [draw=white!80.0!black, fill opacity=0]'
    )  # remove lines that are created for some reason
    tikz_code = tikz_code.replace(
        '(current bounding box.south west)!0.98!(current bounding box.north west)',
        '(current bounding box.south west)!1.05!(current bounding box.north west)'
    )  # position title higher
    tikz_code = tikz_code.replace('title={',
                                  'title={\small ')  # shrink title size
    tikz_code = tikz_code.replace(
        'group style={group size=4 by 4',
        'group style={group size=4 by 4, horizontal sep=1cm, vertical sep=0.4cm '
    )  # reduce separation between plots
    tikz_code = tikz_code.replace(
        'ylabel={Test Loss}', r'ylabel style={align=left}, ylabel=Test\\Loss'
    )  # y label in two lines
    tikz_code = tikz_code.replace(
        'ylabel={Test Accuracy}',
        r'ylabel style={align=left}, ylabel=Test\\Accuracy'
    )  # y label in two lines
    tikz_code = tikz_code.replace(
        'ylabel={Train Loss}', r'ylabel style={align=left}, ylabel=Train\\Loss'
    )  # y label in two lines
    tikz_code = tikz_code.replace(
        'ylabel={Train Accuracy}',
        r'ylabel style={align=left}, ylabel=Train\\Accuracy'
    )  # y label in two lines

    # Write the file out again
    with open(file_name, 'w') as file:
        file.write(tikz_code)

    return tikz_code


def beautify_plot_table(bm_table):
    """Beautify a performance table.

    This function makes a few changes to the performance table to make it nicer.

    Args:
        bm_table (dict): Dictionary holding all the information for the
            performance table.

    Returns:
        pandas.dataframe: A pandas data frame for the performance table.
    """
    bm_table_pd = pd.DataFrame.from_dict({(i, j): bm_table[i][j]
                                          for i in bm_table.keys()
                                          for j in bm_table[i].keys()}).T
    cols = list(bm_table_pd.columns.values)
    if 'AdamOptimizer' in cols:
        cols.insert(0, cols.pop(cols.index('AdamOptimizer')))
    if 'MomentumOptimizer' in cols:
        cols.insert(0, cols.pop(cols.index('MomentumOptimizer')))
    if 'GradientDescentOptimizer' in cols:
        cols.insert(0, cols.pop(cols.index('GradientDescentOptimizer')))
    bm_table_pd = bm_table_pd.reindex(columns=cols)
    print(bm_table_pd)
    return bm_table_pd


def texify_plot_table(perf_table_pd, problem_set):
    """Write a ``.tex`` file with the performance table.

    The function will create a file named `performance_table_small.tex` or
    `performance_table_large.tex` with the latex code for the performance table.

    Args:
        perf_table_pd (pandas.dataframe): Pandas data frame for the performance
            table.
        problem_set (str): Can either be ``small`` or ``large`` to switch
            between which benchmark set is being plotted.

    Returns:
        str: String of the latex code for the performance table.

    """
    if not perf_table_pd.empty:
        # Postprocessing for Latex Output
        pd.set_option('display.max_colwidth', -1)
        perf_table_pd_n = perf_table_pd.apply(
            norm, axis=1)  # normalize between 0 and 100
        perf_table_pd_n_str = perf_table_pd_n.applymap(
            add_color_coding_tex) + perf_table_pd.applymap(
                latex)  # combine normalise version with latex color code command
        perf_table_pd_n_str.columns = perf_table_pd_n_str.columns.str.replace(
            '_', r'\_')  # Texify the column headers
        tikz_code = r"\def\cca#1#2{\cellcolor{green!#1!red}\ifnum #1<50\color{white}\fi{#2}}" +\
        "\n" + r"\resizebox{\textwidth}{!}{%" + "\n" +\
        perf_table_pd_n_str.to_latex(escape=False) + r"}"
        with open('performance_table_' + problem_set + '.tex', 'w') as tex_file:
            tex_file.write(tikz_code)

        return tikz_code


def norm(x):
    """Normalize the input of x, depending on the name (higher is better if
    test_acc is used, otherwise lower is better)"""
    if x.name[1] == 'Tuneability':
        return x
    if x.min() == x.max():
        return x - x.min() + 50.0
    if x.name[1] == 'Performance':
        if x.name[0] == "quadratic_deep" or x.name[0] == "mnist_vae" or x.name[
                0] == "fmnist_vae":
            return np.abs((x - x.max()) / (x.min() - x.max()) * 100)
        else:
            return np.abs((x - x.min()) / (x.max() - x.min()) * 100)
    else:
        return np.abs((x - x.max()) / (x.min() - x.max()) * 100)


def latex(input):
    """Create the latex output version of the input."""
    if isinstance(input, float):
        input = "%.4f" % input
        return "{" + str(input) + "}"
    elif isinstance(input, int):
        return "{" + str(input) + "}"
    elif isinstance(input, dict):
        return str(input).replace('{', '').replace('}', '').replace(
            "'", '').replace('_', '')
    else:
        return ""


def add_color_coding_tex(input):
    """Adds the latex command for color coding to the input"""
    if isinstance(input, str) or isinstance(input, int) or isinstance(
            input, float) and not np.isnan(input):
        return "\cca{" + str(int(input)) + "}"
    else:
        return ""
