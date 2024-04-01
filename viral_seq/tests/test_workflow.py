from viral_seq import run_workflow as workflow
from matplotlib.testing.decorators import image_comparison
import numpy as np


@image_comparison(
    baseline_images=["test_optimization_plotting"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_optimization_plotting(tmp_path):
    rng = np.random.default_rng(seed=2024)
    data = {
        "Classifier1": [rng.uniform() for i in range(30)],
        "Classifier2": [rng.uniform() for i in range(10)],
        "Classifier3": [rng.uniform() for i in range(51)],
    }
    workflow.optimization_plots(
        data,
        str(tmp_path / "test_optimization_plotting.csv"),
        str(tmp_path / "test_optimization_plotting.png"),
    )
