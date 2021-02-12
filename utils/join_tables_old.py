import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

MEAN_TABLES = ['tables/Test/Test_ff_model_predictions_state_threshold_table.csv',
               'tables/Test/Test_ff_model_dpo_predictions_state_threshold_table.csv',
               'tables/Test/Test_ff_model_ta_predictions_state_threshold_table.csv',
               'tables/Test/Test_ff_model_dpo_ta_predictions_state_threshold_table.csv',
               'tables/Test/Test_gru_model_predictions_state_threshold_table.csv',
               'tables/Test/Test_gru_model_dpo_predictions_state_threshold_table.csv',
               'tables/Test/Test_gru_model_ta_predictions_state_threshold_table.csv',
               'tables/Test/Test_gru_model_dpo_ta_predictions_state_threshold_table.csv',
               'tables/Test/Test_lstm_model_predictions_state_threshold_table.csv',
               'tables/Test/Test_lstm_model_dpo_predictions_state_threshold_table.csv',
               'tables/Test/Test_lstm_model_ta_predictions_state_threshold_table.csv',
               'tables/Test/Test_lstm_model_dpo_ta_predictions_state_threshold_table.csv']

STDDEV_TABLES = ['tables/Test/Test_ff_model_predictions_state_threshold_table_stddev.csv',
                 'tables/Test/Test_ff_model_dpo_predictions_state_threshold_table_stddev.csv',
                 'tables/Test/Test_ff_model_ta_predictions_state_threshold_table_stddev.csv',
                 'tables/Test/Test_ff_model_dpo_ta_predictions_state_threshold_table_stddev.csv',
                 'tables/Test/Test_gru_model_predictions_state_threshold_table_stddev.csv',
                 'tables/Test/Test_gru_model_dpo_predictions_state_threshold_table_stddev.csv',
                 'tables/Test/Test_gru_model_ta_predictions_state_threshold_table_stddev.csv',
                 'tables/Test/Test_gru_model_dpo_ta_predictions_state_threshold_table_stddev.csv',
                 'tables/Test/Test_lstm_model_predictions_state_threshold_table_stddev.csv',
                 'tables/Test/Test_lstm_model_dpo_predictions_state_threshold_table_stddev.csv',
                 'tables/Test/Test_lstm_model_ta_predictions_state_threshold_table_stddev.csv',
                 'tables/Test/Test_lstm_model_dpo_ta_predictions_state_threshold_table_stddev.csv']

def build_table(tables, cutoff):
    columns = ["Model", "Metric", "20%", "40%", "60%", "80%"]
    data = []
    for table_file in tables:
        tab = pd.read_csv(table_file, header = 0, index_col = 0)
        terms = table_file.split('_')
        model_name = '_'.join([terms[1]] + terms[3:cutoff])
        for i in range(2):
            add_to_data = [model_name]
            if i == 0:
                add_to_data.append('D_on')
            elif i == 1:
                add_to_data.append('D_off')
            for j in range(4):
                add_to_data.append(round(tab.loc['All'].iloc[5 + j*2 + i], 3))
            data.append(add_to_data)
    return pd.DataFrame(data, columns = columns)

def rank_models(mean_tab, std_tab, metric):
    results = defaultdict(list)
    mean_tab = mean_tab[mean_tab["Metric"] == metric]
    std_tab = std_tab[std_tab["Metric"] == metric]
    for threshold in ("20%", "40%", "60%", "80%"):
        mean_sorted = mean_tab[threshold].abs().argsort()
        std_sorted = std_tab[threshold].argsort()
        mean_seen = set()
        std_seen = set()
        for i in range(len(mean_sorted)):
            mean_val = mean_tab["Model"].iloc[mean_sorted.iloc[i]]
            std_val = std_tab["Model"].iloc[std_sorted.iloc[i]]
            if mean_val in std_seen:
                results[threshold].append(mean_val)
            mean_seen.add(mean_val)
            if std_val in mean_seen:
                results[threshold].append(std_val)
            std_seen.add(std_val)
    return results

def reverse_rank(rank):
    results = defaultdict(list)
    for threshold in ("20%", "40%", "60%", "80%"):
        for i in range(len(rank[threshold])):
            results[rank[threshold][i]].append(12 - i)
    return results

def plot_rank(rank, metric):
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)
    for model in sorted(rank):
        plt.plot([20, 40, 60, 80], rank[model], label = model)
    plt.title("Model Ranks for " + metric + " Metric")
    plt.yticks(list(range(1, 13))[::1], list(range(1, 13))[::-1])
    plt.xticks([20, 40, 60, 80])
    plt.legend()
    plt.show()

def main():
    mean_tab = build_table(MEAN_TABLES, -4)
    std_tab = build_table(STDDEV_TABLES, -5)

    on_rank = rank_models(mean_tab, std_tab, "D_on")
    off_rank = rank_models(mean_tab, std_tab, "D_off")

    on_rev, off_rev = reverse_rank(on_rank), reverse_rank(off_rank)

    plot_rank(on_rev, "$D_{on}$")
    plot_rank(off_rev, "$D_{off}$")

    mean_tab.to_csv('tables/mean_threshold_table.csv')
    std_tab.to_csv('tables/stddev_threshold_table.csv')

if __name__ =='__main__':
    main()