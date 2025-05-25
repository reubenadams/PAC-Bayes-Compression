import matplotlib.pyplot as plt
import pandas as pd

from config import ComplexityMeasures



pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)


def main():
    target_CE_loss_increase = 0.1
    combined_df = pd.read_csv(r"distillation\models\MNIST1D\40\all_metrics\combined.csv")
    complexity_measure_names = ComplexityMeasures.get_all_names(target_CE_loss_increase=target_CE_loss_increase)

    y = combined_df["Base Generalization Gap"]
    for name in complexity_measure_names:
        matplotlib_name = ComplexityMeasures.get_matplotlib_name(name, target_CE_loss_increase=target_CE_loss_increase)
        x = combined_df[name]
        plt.scatter(x, y, s=2)
        plt.xlabel(matplotlib_name, fontsize=12)
        plt.ylabel(r"Generalization Gap, $R_D(h_{W, B})-R_S(h_{W, B})$", fontsize=12)
        if ComplexityMeasures.use_log_x_axis(name):
            plt.xscale("log")
        plt.show()


if __name__ == "__main__":
    main()