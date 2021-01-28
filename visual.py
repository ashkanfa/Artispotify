"""Functions used for visualizing data."""


# Import libraries and functions
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns; sns.set()
from spotify_API import *
from ML_tool import *



def log10ticks(y, pos):
    """Scientific notation for tick marks via FuncFormatter()."""
    return '$10^{:.0f}$'.format(y)



def rebin(data, binwidth):
    """Bin the data by binwidth."""
    binning = np.arange(min(data), max(data) + binwidth, binwidth)
    return data, binning



def plot_columns(input_df):
    """Plot distributions of all feature columns from a dataframe."""
    # Drop the irrelevant columns
    all_features = drop_cols(input_df)

    # Plot the distributions and format the plots
    fig, axs = plt.subplots(len(all_features.columns), figsize=(14, 90))
    for num, col in enumerate(all_features.columns):
        plt.sca(axs[num])
        dist = sns.distplot(all_features[col], kde=False)
        plt.xlabel(col.replace('Track_', ''), fontsize=18)
        plt.xticks(fontsize=14)
        plt.ylabel('# of Tracks', fontsize=18)
        plt.yticks(fontsize=14)
    plt.show()



def plot_popularity(input_df):
    """Plot the distribution of track popularity."""
    # Plot the data and format the plot
    plt.figure(figsize=(14, 6))
    dist = sns.distplot(input_df['Track_Popularity'], kde=False, bins=np.array(range(0,100,2)))
    plt.xlabel('Popularity Score', fontsize=18)
    plt.xlim(0, 100)
    plt.xticks(fontsize=14)
    plt.ylabel('# of Tracks', fontsize=18)
    plt.yticks(fontsize=14)
    plt.show()



def plot_correlations(input_df):
    """Plot the correlations of the dataframe features."""
    # Drop the irrelevant columns
    all_features = drop_cols(input_df)

    # Get the correlations and set the column names
    corr = all_features.corr()
    col_names = [x.replace('Track_', '') for x in all_features.columns]

    # Plot the correlation heatmap and format the plot
    plt.figure(figsize=(10, 8))
    htmp = sns.heatmap(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    htmp.set_xticklabels(col_names, fontsize=14)
    htmp.set_yticklabels(col_names, fontsize=14)
    htmp.collections[0].colorbar.ax.tick_params(labelsize=14)
    plt.show()



def plot_follower_count(filerange=range(201), plot_legend=False):
    """Plot the distribution of followers from the evaluation sample of artists."""
    # Load the data
    results_list = load_sample_data(filerange)

    # Get the follower counts from the evaluation sample
    followers = []
    for res in results_list:
        followers.append(res[1][2])

    # Get the follower counts from the full 2000 random artist list for comparison
    fpath = 'Data/random_artists.pkl'
    with open(fpath, 'rb') as f:
        random_artists = pickle.load(f)
    all_followers = [x[2] for x in random_artists]

    # Set the bins for the histograms
    data_full, binning = rebin(np.log10(np.array(all_followers) + 1), 0.25) # +1 in case of log(0)
    data_sample, binning = rebin(np.log10(np.array(followers) + 1), 0.25) # +1 in case of log(0)

    # Plot the two distributions together and format the plot
    plt.figure(figsize=(6, 6))
    dist = sns.distplot(data_full, bins=binning, label='2000 random artists', kde=True)
    dist = sns.distplot(data_sample, bins=binning, label=r'10% subsample', kde=True)
    plt.xlabel('# of Followers', fontsize=18)
    plt.xticks(fontsize=14)
    dist.axes.xaxis.set_major_formatter(mtick.FuncFormatter(log10ticks))
    plt.ylabel('# of Artists', fontsize=18)
    plt.yticks(fontsize=14)
    if plot_legend:
        plt.legend(loc='upper left', fontsize=14)
    plt.show()



def plot_network_sizes(filerange=range(201)):
    """Plot the distribution of network sizes in the evaluation sample of artists."""
    # Load the data
    results_list = load_sample_data(filerange)

    # Pull out the length of each network
    all_nets = []
    for res in results_list:
        all_nets.append(len(res[2][1]))

    # Set the bins for the histogram
    data, binning = rebin(all_nets, 25)

    # Plot the data and format the plot
    plt.figure(figsize=(6, 6))
    dist = sns.distplot(data, bins=binning, kde=False)
    plt.xlabel('# of Artists in Network', fontsize=18)
    plt.xticks(fontsize=14)
    plt.ylabel('# of Artists', fontsize=18)
    plt.yticks(fontsize=14)
    plt.show()



def plot_tracklist_sizes(filerange=range(201)):
    """Plot the distribution of seed artist tracklist sizes in the evaluation sample of artists."""
    # Load the data
    results_list = load_sample_data(filerange)

    # Pull out the length of each track list
    all_tracklists = []
    for res in results_list:
        all_tracklists.append(len(res[2][2]))

    # Set the bins for the histogram
    data, binning = rebin(np.log10(np.array(all_tracklists) + 1), 0.25) # +1 in case of log(0)

    # Plot the data and format the plot
    plt.figure(figsize=(6, 6))
    dist = sns.distplot(data, bins=binning, kde=False)
    plt.xlabel('# of Tracks in Artist\'s Library', fontsize=18)
    plt.xticks(fontsize=14)
    dist.axes.xaxis.set_major_formatter(mtick.FuncFormatter(log10ticks))
    plt.ylabel('# of Artists', fontsize=18)
    plt.yticks(fontsize=14)
    plt.show()



def plot_reclist_sizes(filerange=range(201)):
    """Plot the distribution of recommended tracklist sizes in the evaluation sample of artists."""
    # Load the data
    results_list = load_sample_data(filerange)

    # Pull out the length of each recommendation list
    all_recs = []
    for res in results_list:
        all_recs.append(len(res[2][4]))

    # Set the bins for the histogram
    data, binning = rebin(all_recs, 1000)

    # Plot the data and format the plot
    plt.figure(figsize=(6, 6))
    dist = sns.distplot(data, bins=binning, kde=False)
    plt.xlabel('# of Tracks in Recommendation List', fontsize=18)
    plt.xticks(fontsize=14)
    plt.ylabel('# of Artists', fontsize=18)
    plt.yticks(fontsize=14)
    plt.show()



def plot_tuning_curve(mean_dict, std_dict, param_vals,
                 label_vals=['', 'Parameter Value', 'Score'],
                 color_vals=['black'],
                 ymax=None, logx=False):
    """Plot curves for hyperparameter turning of models.

    mean_dict - dictionary of mean values (each key is a different curve)
    std_dict - dictionary of standard deviation values (each key is a different curve)
    param_vals - the range of values for the hyperparameter (i.e. the x-axis values)
    label_vals - labels for the title, x-axis, and y-axis
    color_vals - values for the colors of the different curves, in order of columns
    ymax - if set, will make the y-axis limits (0, ymax)
    logx - if True, will use a log scale on the x-axis
    """
    # Set up the figure
    plt.figure(figsize=(16, 8))

    # Iterate through the different curves to plot
    for n, key in enumerate(mean_dict):
        # Set the color of the curve (default to black)
        if n < len(color_vals):
            color_choice = color_vals[n]
        else:
            color_choice = 'black'

        # Plot the means
        m = np.array(mean_dict[key])
        if logx:
            plt.semilogx(param_vals, m, label=key, color=color_choice, lw=2)
        else:
            plt.plot(param_vals, m, label=key, color=color_choice, lw=2)

        # Add the standard deviations
        if key in std_dict:
            s = np.array(std_dict[key])
            plt.fill_between(param_vals, m-s, m+s, alpha=0.2, color=color_choice, lw=1)

    # Format and display the plot
    plt.title(label_vals[0], fontsize=20)
    plt.legend(loc='best', fontsize=14)
    plt.xlabel(label_vals[1], fontsize=18)
    if logx:
        plt.xlim(param_vals[0] / 10, xmax = param_vals[-1] * 10)
    else:
        plt.xlim(0, param_vals[0] + param_vals[-1])
    plt.xticks(fontsize=14)
    plt.ylabel(label_vals[2], fontsize=18)
    if ymax:
        plt.ylim(0, ymax)
    plt.yticks(fontsize=14)
    plt.show()



def print_RFC_importances(sorted_mean, sorted_labels):
    """Print the sign and magnitude for the random forest classifier feature importances.

    sorted_mean - the average value of importance for the different features
    sorted_labels - the names of the different features
    """
    # List the features by magnitude of importance
    for n in range(len(sorted_labels)):
        if sorted_mean[n] < 0:
            msg = '-'
        else:
            msg = '+'
        print('{:>16}:  {} {:4.1f}%'.format(sorted_labels[n], msg, abs(sorted_mean[n]*100)))



def plot_RFC_importances(sorted_mean, sorted_std, sorted_labels, sorted_colors, st_xlabels=False):
    """Plot the random forest classifier feature importances.
    Also returns a dataframe of the results for Streamlit.

    sorted_mean - the average value of importance for the different features
    sorted_std - the standard deviation of importance for the different features
    sorted_labels - the names of the different features
    sorted_colors - colors to use for the bar graph
    st_xlabels - if True, change the xticks to integers instead of names
    """
    # Create a dataframe of the results
    mean_disp_list = []
    msg_list = []
    for imp in sorted_mean:
        if imp < 0:
            mean_disp = '- {:4.1f}%'.format(abs(imp)*100)
            msg = 'Drives it DOWN'
        else:
            mean_disp = '+ {:4.1f}%'.format(imp*100)
            msg = 'Drives it UP'
        mean_disp_list.append(mean_disp)
        msg_list.append(msg)
    importances = pd.DataFrame({'Feature':sorted_labels, 'Impact on Popularity':msg_list, 'Importance':mean_disp_list})

    # Plot the data and format the plot
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.bar(range(len(sorted_labels)), sorted_mean, yerr=sorted_std, color=sorted_colors)
    ax.set_title('How Audio Features Drive Popularity', fontsize=14, fontweight="bold", 'horizontalalignment'='center')

    if st_xlabels:
        ax.set_xticks(range(len(sorted_labels)))
        ax.set_xticklabels(range(1, len(sorted_labels) + 1))
        #ax.xticks(range(len(sorted_labels)), range(1, len(sorted_labels) + 1), fontsize=12)
        ax.set_xlabel('Feature # (see below)', fontsize=13)
    else:
        ax.set_xticks(range(len(sorted_labels)))
        ax.set_xticklabels(sorted_labels)
        ax.set_xlabel('Audio Features', fontsize=13)
        #ax.xticks(range(len(sorted_labels)), sorted_labels, rotation=75, fontsize=16)

    ax.set_ylabel('Relative Importance', fontsize=13)
    #ax.yticks(fontsize=14)
    #plt.show()
    return importances, fig
