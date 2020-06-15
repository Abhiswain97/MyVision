import csv
import os

if not os.path.exists("logs"):
    os.mkdir("logs")


class CSVLogger:
    def __call__(self, metrics,  metric_name):
        with open(os.path.join("logs", "metrics.csv"), mode="w", newline='') as metrics_csv:
            csv_writer = csv.writer(
                metrics_csv, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(
                ["Epoch", "Training loss", "Validation loss", metric_name]
            )

            csv_writer.writerows(
                metrics
            )
