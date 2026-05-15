"""
Script version of `inference/Graph sequence model.ipynb`.

Goal: run the same steps as the notebook (training, evaluation, and figure generation),
but from the command line.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import random
import string
import subprocess
import sys
from pathlib import Path

# Match notebook behavior: disable W&B if present.
os.environ.setdefault("WANDB_MODE", "disabled")


INFERENCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INFERENCE_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _add_repo_import_paths() -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))


_add_repo_import_paths()

class GPT:
    def __init__(self, base_model: str | None = None, base_model_name: str = "gpt2", vocab_size: int = 100):
        self.base_model = base_model
        self.base_model_name = base_model_name
        self.vocab_size = vocab_size

        if self.base_model is not None:
            try:
                from transformers import GPT2LMHeadModel, GPT2TokenizerFast
            except ModuleNotFoundError as e:  # pragma: no cover
                raise ModuleNotFoundError(
                    "Missing dependency `transformers`. Install with `pip install -r requirements.txt`."
                ) from e
            self.tokenizer = GPT2TokenizerFast.from_pretrained(base_model)
            self.model = GPT2LMHeadModel.from_pretrained(base_model)
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def continue_input(
        self,
        input_sequence: str,
        max_new_tokens: int = 5,
        num_return_sequences: int = 1,
        no_repeat_ngram_size: int = 0,
        do_sample: bool = False,
        temperature: float = 0.7,
        num_beams: int = 1,
    ) -> str:
        input_ids = self.tokenizer.encode(input_sequence, return_tensors="pt")

        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            temperature=temperature,
        )

        sequence = output[0].tolist()
        text = self.tokenizer.decode(sequence)
        return text


def load_pkl(pth: str) -> object:
    with open(pth, "rb") as f:
        return pickle.load(f)


def is_valid_path(sequence: str, graphs) -> bool:
    parts = sequence.split()
    nodes = parts[::2]

    for graph in graphs:
        path_exists = True
        for i in range(len(nodes) - 1):
            if not graph.has_edge(nodes[i], nodes[i + 1]):
                path_exists = False
                break
        if path_exists:
            return True
    return False


def _run_run_clm(
    *,
    train_path: str,
    test_path: str,
    output_dir: str,
    num_epochs: int,
    lr: float,
    tokenizer_name: str,
) -> None:
    run_clm = SCRIPTS_DIR / "run_clm.py"
    cmd = [
        sys.executable,
        str(run_clm),
        "--model_type",
        "gpt2",
        "--tokenizer_name",
        tokenizer_name,
        "--config_name",
        "gpt2-medium",
        "--train_file",
        train_path,
        "--validation_file",
        test_path,
        "--per_device_train_batch_size",
        "1",
        "--per_device_eval_batch_size",
        "1",
        "--do_train",
        "--do_eval",
        "--output_dir",
        output_dir,
        "--overwrite_output_dir",
        "--num_train_epochs",
        str(num_epochs),
        "--save_strategy",
        "epoch",
        "--eval_strategy",
        "steps",
        "--eval_steps",
        "2000",
        "--learning_rate",
        str(lr),
    ]
    subprocess.run(cmd, cwd=str(INFERENCE_DIR), check=True)


def train_model_script(
    *,
    num_epochs: int = 3,
    output_dir: str = "outputs",
    lr: float = 5e-05,
    tokenizer_name: str,
) -> None:
    gc.collect()
    train_path = f"./{output_dir}/train.txt"
    test_path = f"./{output_dir}/test.txt"
    _run_run_clm(
        train_path=train_path,
        test_path=test_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        lr=lr,
        tokenizer_name=tokenizer_name,
    )


def _rm_rf(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        for child in path.iterdir():
            _rm_rf(child)
        path.rmdir()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _train_spatial(
    *,
    n_graphs_train: int,
    n_walks_train: int,
    max_walk_length: int,
    n_graphs_test: int,
    n_walks_test: int,
) -> None:
    try:
        from graph_utils import generate_name, generate_n_random_walks, get_graph, get_walks_as_strings
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency needed by `scripts/graph_utils.py` (likely `csrgraph`). "
            "Install with `pip install -r requirements.txt`."
        ) from e

    out_dir = INFERENCE_DIR / "outputs_graph"
    _rm_rf(out_dir) if out_dir.exists() else None
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training: n_walks_train walks per graph, each with random length in [1, max_walk_length]
    walks = []
    train_gs = []
    for _ in range(n_graphs_train):
        nodes = [generate_name() for _ in range(9)]
        G = get_graph(nodes=nodes)
        train_gs.append(G)
        for _ in range(n_walks_train):
            wl = random.randint(1, max_walk_length)
            walk = generate_n_random_walks(G, n_walks=1, walk_length=wl)[0]
            walks.append(walk)
    _write_text(out_dir / "train.txt", "\n".join(walks))

    walks, test_gs = get_walks_as_strings(n_graphs=n_graphs_test, n_walks=n_walks_test, walk_length=max_walk_length)
    _write_text(out_dir / "test.txt", "\n".join(walks))

    tokenizer_dir = ensure_entity_tokenizer_dir()
    train_model_script(num_epochs=1, output_dir="outputs_graph", lr=5e-05, tokenizer_name=str(tokenizer_dir))

    with open(out_dir / "train_graphs.pkl", "wb") as handle:
        pickle.dump(train_gs, handle)
    with open(out_dir / "test_graphs.pkl", "wb") as handle:
        pickle.dump(test_gs, handle)


def _train_family(
    *,
    n_graphs_train: int,
    n_walks_train: int,
    max_walk_length: int,
    n_graphs_test: int,
    n_walks_test: int,
) -> None:
    try:
        from tree_utils import (
            create_extended_family_tree,
            create_family_tree_digraph,
            generate_random_walks,
            get_walks_for_n_trees,
            infer_grandparent_edges,
        )
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency needed by `scripts/tree_utils.py` (likely `csrgraph`). "
            "Install with `pip install -r requirements.txt`."
        ) from e

    out_dir = INFERENCE_DIR / "outputs_tree"
    _rm_rf(out_dir) if out_dir.exists() else None
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training: n_walks_train walks per tree, each with random length in [1, max_walk_length]
    walks = []
    train_gs = []
    for _ in range(n_graphs_train):
        relationships = create_extended_family_tree(base_num_children=2, grandparent_num_children=2)
        relationships = infer_grandparent_edges(relationships)
        G = create_family_tree_digraph(relationships)
        train_gs.append(G)
        for _ in range(n_walks_train):
            wl = random.randint(1, max_walk_length)
            walk = generate_random_walks(G, n=1, walk_length=wl)[0]
            walks.append(walk)
    _write_text(out_dir / "train.txt", "\n".join(walks))

    walks, test_gs = get_walks_for_n_trees(n_graphs=n_graphs_test, n_walks=n_walks_test, walk_length=max_walk_length)
    _write_text(out_dir / "test.txt", "\n".join(walks))

    tokenizer_dir = ensure_entity_tokenizer_dir()
    train_model_script(num_epochs=1, output_dir="outputs_tree", lr=5e-05, tokenizer_name=str(tokenizer_dir))

    with open(out_dir / "train_trees.pkl", "wb") as handle:
        pickle.dump(train_gs, handle)
    with open(out_dir / "test_trees.pkl", "wb") as handle:
        pickle.dump(test_gs, handle)


def _generate_name() -> str:
    return "".join(random.choices(string.ascii_lowercase, k=2))


def ensure_entity_tokenizer_dir() -> Path:
    """
    Build (once) a tokenizer that treats 2-letter entities as single tokens.

    The tokenizer is saved under `inference/tokenizers/`, and its path is passed
    to `scripts/run_clm.py` via `--tokenizer_name`.
    """
    tokenizer_dir = INFERENCE_DIR / "tokenizers" / "gpt2-medium_2letter_entities"
    if (tokenizer_dir / "tokenizer.json").exists() or (tokenizer_dir / "vocab.json").exists():
        return tokenizer_dir

    try:
        from entity_tokenizer import build_and_save_entity_tokenizer
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing `scripts/entity_tokenizer.py` import. Ensure `scripts/` is on sys.path."
        ) from e

    tokenizer_dir.parent.mkdir(parents=True, exist_ok=True)
    return build_and_save_entity_tokenizer(output_dir=tokenizer_dir)


def test_loop_greedy(
    model: GPT,
    loop_templates: list[str | tuple[str, list[int]]],
) -> tuple[float, dict[str, float]]:
    accuracy_scores: list[int] = []
    results_dict: dict[str, float] = {}

    for item in loop_templates:
        if isinstance(item, tuple):
            template, entity_indices = item
            n_unique = max(entity_indices) + 1
        else:
            template = item
            entity_indices = None

        template_accuracy: list[int] = []

        for _ in range(100):
            if entity_indices is not None:
                names = [_generate_name() for _ in range(n_unique)]
                fill_args = [names[i] for i in entity_indices]
                filled_template = template.format(*fill_args)
                true_final_item = fill_args[-1]
            else:
                names = [_generate_name() for _ in range(template.count("{}") - 1)]
                names += [names[0]]
                filled_template = template.format(*names)
                true_final_item = names[-1]

            print(filled_template)

            input_len = len(filled_template.split())

            prediction = model.continue_input(filled_template[0:-3], max_new_tokens=5, do_sample=False)
            print(prediction)

            predicted_items = prediction.strip().split()[0:input_len]
            predicted_final_item = predicted_items[-1] if predicted_items else None
            print(f"True final:{true_final_item}, predicted final: {predicted_final_item}")

            is_correct = int(predicted_final_item == true_final_item)
            print(is_correct)
            template_accuracy.append(is_correct)

        accuracy_scores.extend(template_accuracy)
        results_dict[template] = sum(template_accuracy) / len(template_accuracy)

    overall_avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    return overall_avg_accuracy, results_dict


def _plot_aggregated_inf(*, family_results_dict: dict[str, float], spatial_results_dict: dict[str, float]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    def get_hops_count(key: str) -> int:
        return len(key.split()) // 2

    combined_data = {"Family tree": family_results_dict, "Spatial": spatial_results_dict}
    averages: dict[int, dict[str, list[float] | float]] = {}
    std_devs: dict[int, dict[str, float]] = {}

    for task, data in combined_data.items():
        for pattern, accuracy in data.items():
            hops = get_hops_count(pattern)
            if hops not in averages:
                averages[hops] = {}
                std_devs[hops] = {}
            if task not in averages[hops]:
                averages[hops][task] = []
            assert isinstance(averages[hops][task], list)
            averages[hops][task].append(accuracy)

    for hops, tasks in averages.items():
        for task, accuracies in tasks.items():
            assert isinstance(accuracies, list)
            averages[hops][task] = float(np.mean(accuracies))
            std_devs[hops][task] = float(np.std(accuracies))

    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    tasks = list(combined_data.keys())
    colors = ["blue", "red"]
    hops_labels = sorted(averages.keys())

    x = np.arange(len(hops_labels))
    bar_width = 0.35
    offset = bar_width / 2

    for i, hops in enumerate(hops_labels):
        positions = x[i] - offset * len(tasks) / 2
        for j, task in enumerate(tasks):
            avg = float(averages[hops].get(task, 0))  # type: ignore[arg-type]
            std_dev = float(std_devs[hops].get(task, 0))
            bar_pos = positions + j * bar_width
            ax.bar(
                bar_pos,
                avg,
                bar_width,
                label=task if i == 0 else "",
                color=colors[j],
                alpha=0.4,
                yerr=std_dev,
                capsize=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} hops" for h in hops_labels])
    ax.set_xlabel("Number of transitions")
    ax.set_ylabel("Average accuracy")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.12)

    plt.tight_layout()
    plt.savefig("aggregated_inf.png", dpi=300)
    plt.show()


def _plot_loss(trainer_state_path: Path, out_png: str) -> None:
    import matplotlib.pyplot as plt

    trainer_state = json.loads(trainer_state_path.read_text())

    train_steps: list[float] = []
    train_loss: list[float] = []
    eval_steps: list[float] = []
    eval_loss: list[float] = []

    for entry in trainer_state["log_history"]:
        if "loss" in entry:
            train_steps.append(entry["epoch"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["epoch"])
            eval_loss.append(entry["eval_loss"])

    plt.figure(figsize=(2, 2))
    plt.plot(train_steps, train_loss, label="Train loss", marker=".", color="red", markersize=1, alpha=0.3)
    plt.plot(eval_steps, eval_loss, label="Val loss", marker=".", color="blue", markersize=1, alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()


def track_coordinates(walks: list[str]) -> bool:
    direction_offsets = {"NORTH": (0, 1), "SOUTH": (0, -1), "EAST": (1, 0), "WEST": (-1, 0)}

    coordinates: dict[tuple[int, int], str] = {}
    current_position = (0, 0)

    for walk in walks:
        steps = walk.split()
        for i in range(0, len(steps) - 2, 2):
            node = steps[i]
            direction = steps[i + 1]
            next_node = steps[i + 2]

            if current_position in coordinates:
                if coordinates[current_position] != node:
                    print(
                        f"Invalid path: {node} found at {current_position}, but {coordinates[current_position]} was expected."
                    )
                    return False
            else:
                coordinates[current_position] = node

            if direction not in direction_offsets.keys():
                return False
            else:
                offset = direction_offsets[direction]
                current_position = (current_position[0] + offset[0], current_position[1] + offset[1])

                if current_position in coordinates:
                    if coordinates[current_position] != next_node:
                        return False
                else:
                    coordinates[current_position] = next_node

    return True


def _random_letter_pair() -> str:
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=2))


def _imagination(spatial_model_path: str) -> dict[float, list[str]]:
    import matplotlib.pyplot as plt
    import numpy as np

    model = GPT(base_model=spatial_model_path, base_model_name="gpt2")

    imagined_for_temps: dict[float, list[str]] = {}
    for temp in [0, 0.5, 1.0, 1.5, 2.0]:
        imagined: list[str] = []
        for _ in range(50):
            if temp == 0:
                prediction = model.continue_input(_random_letter_pair(), do_sample=False, max_new_tokens=50)
            else:
                prediction = model.continue_input(
                    _random_letter_pair(), do_sample=True, max_new_tokens=50, temperature=temp
                )
            imagined.append(prediction)
        imagined_for_temps[temp] = imagined

    lengths = [1, 2, 3, 4, 5, 6]
    plt.figure(figsize=(2, 2.5))

    cmap = plt.get_cmap("magma")
    colors = cmap(np.linspace(0.15, 0.75, len(imagined_for_temps)))

    temps_to_plot = [0.5, 1.0, 1.5, 2.0]
    for idx, temp in enumerate(temps_to_plot):
        paths = imagined_for_temps[temp]
        fractions: list[float] = []
        for length in lengths:
            valid_count = 0
            for path in paths:
                shortened_path = " ".join(path.split()[: 2 * length + 1])
                if track_coordinates([shortened_path]):
                    valid_count += 1
            fraction_valid = valid_count / len(paths)
            fractions.append(fraction_valid)
        plt.plot(lengths, fractions, marker="o", label=f"{temp}", color=colors[idx])

    plt.xlabel("Number of transitions")
    plt.ylabel("Fraction valid")
    plt.legend(title="Temp.")
    plt.savefig("Imagined_paths_by_temp.png", dpi=300, bbox_inches="tight")
    plt.show()

    import seaborn as sns

    def path_to_coordinates(path: str):
        x, y = 0, 0
        coordinates = [(x, y)]
        directions = {"NORTH": (0, 1), "SOUTH": (0, -1), "EAST": (1, 0), "WEST": (-1, 0)}
        steps = path.split()
        for step in steps:
            if step in directions:
                dx, dy = directions[step]
                x += dx
                y += dy
                coordinates.append((x, y))
        return coordinates

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    grid_size = 9
    center = grid_size // 2

    temps_to_plot = [0, 0.5, 1.0]
    for idx, temp in enumerate(temps_to_plot):
        paths = imagined_for_temps[temp]
        grid = np.zeros((grid_size, grid_size))

        for path in paths:
            coordinates = path_to_coordinates(path)
            for x, y in coordinates:
                grid[center + x, center + y] += 1

        sns.heatmap(grid, cmap="coolwarm", cbar=True, ax=axs[idx], vmin=0, vmax=250, alpha=0.7)
        axs[idx].set_title(f"Temperature of {temp}")
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])

    plt.tight_layout()
    plt.savefig("imagined_heatmaps.png", dpi=300)
    plt.show()

    def calculate_max_distance_from_origin(coordinates):
        max_distance = 0
        for (x, y) in coordinates:
            distance = abs(x) + abs(y)
            if distance > max_distance:
                max_distance = distance
        return max_distance

    temps_to_plot = [0, 0.5, 1.0, 1.5, 2.0]
    mean_distances = []
    sem_distances = []
    all_distances = []

    for temp in temps_to_plot:
        distances = []
        paths = imagined_for_temps[temp]

        for path in paths:
            coordinates = path_to_coordinates(path)
            max_distance = calculate_max_distance_from_origin(coordinates)
            distances.append(max_distance)

        mean_distance = np.mean(distances)
        sem_distance = np.std(distances)

        mean_distances.append(mean_distance)
        sem_distances.append(sem_distance)
        all_distances.append(distances)

    plt.figure(figsize=(2, 2.5))
    plt.bar(
        temps_to_plot,
        mean_distances,
        yerr=sem_distances,
        width=0.4,
        capsize=2,
        color="blue",
        alpha=0.4,
        label="Mean Distance",
    )

    for i, temp in enumerate(temps_to_plot):
        x_values = np.full(len(all_distances[i]), temp)
        plt.scatter(x_values, all_distances[i], color="blue", alpha=0.2, label="Individual Distances" if i == 0 else "")

    plt.xlabel("Temperature")
    plt.ylabel("Mean max distance")
    plt.savefig("dists.png", dpi=300, bbox_inches="tight")
    plt.show()

    return imagined_for_temps


def test_loop_sampled(model: GPT, loop_templates: list[str]) -> tuple[float, dict[str, float]]:
    accuracy_scores: list[int] = []
    results_dict: dict[str, float] = {}

    for template in loop_templates:
        template_accuracy: list[int] = []

        for _ in range(50):
            names = [_generate_name() for _ in range(template.count("{}") - 1)]
            names += [names[0]]
            filled_template = template.format(*names)
            print(filled_template)

            true_final_item = names[-1]
            input_len = len(filled_template.split())

            prediction = model.continue_input(
                filled_template[0:-3],
                max_new_tokens=5,
                do_sample=True,
                temperature=1.0,
                num_beams=5,
            )
            print(prediction)
            predicted_items = prediction.strip().split()[0:input_len]
            predicted_final_item = predicted_items[-1] if predicted_items else None
            print(f"True final:{true_final_item}, predicted final: {predicted_final_item}")

            is_correct = int(predicted_final_item == true_final_item)
            print(is_correct)
            template_accuracy.append(is_correct)

        accuracy_scores.extend(template_accuracy)
        results_dict[template] = sum(template_accuracy) / len(template_accuracy)

    overall_avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    return overall_avg_accuracy, results_dict


def create_valid_loop_templates(n: int) -> list[str]:
    directions_tuples = [
        ("EAST", "SOUTH", "WEST", "NORTH"),
        ("NORTH", "EAST", "SOUTH", "WEST"),
        ("WEST", "NORTH", "EAST", "SOUTH"),
        ("SOUTH", "WEST", "NORTH", "EAST"),
    ]

    templates = []
    for direction_tuple in directions_tuples:
        direction_tuple = [[i] * n for i in list(direction_tuple)]
        direction_tuple = [item for sublist in direction_tuple for item in sublist]
        template = " {} ".join(direction_tuple)
        template = "{} " + template + " {}"
        templates.append(template)
    return templates


def create_repetition_templates(m: int) -> list[str]:
    templates = []

    rep_template_1 = " {} ".join(["NORTH"] * m + ["EAST"] + ["SOUTH"] * m + ["WEST"])
    rep_template_2 = " {} ".join(["NORTH"] + ["EAST"] * m + ["SOUTH"] + ["WEST"] * m)
    rep_template_3 = " {} ".join(["NORTH"] + ["WEST"] * m + ["SOUTH"] + ["EAST"] * m)
    rep_template_4 = " {} ".join(["NORTH"] * m + ["WEST"] + ["SOUTH"] * m + ["EAST"])
    templates.append("{} " + rep_template_1 + " {}")
    templates.append("{} " + rep_template_2 + " {}")
    templates.append("{} " + rep_template_3 + " {}")
    templates.append("{} " + rep_template_4 + " {}")

    return templates


def generate_loop_templates(min_n: int = 1, max_n: int = 5) -> dict[int, list[str]]:
    templates_dict: dict[int, list[str]] = {}
    for n in range(min_n, max_n + 1):
        templates = create_valid_loop_templates(n)
        repetition_templates = create_repetition_templates(n)
        templates_dict[n] = templates + repetition_templates
    return templates_dict


def _grid_generalisation(spatial_model_path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    loop_templates_dict = generate_loop_templates()
    model = GPT(base_model=spatial_model_path, base_model_name="gpt2")

    results: dict[int, tuple[float, float]] = {}
    for n, templates in loop_templates_dict.items():
        accuracies = []
        for template in templates:
            accuracy, _ = test_loop_sampled(model, [template])
            accuracies.append(accuracy)
        average_accuracy = float(np.mean(accuracies))
        sem = float(np.std(accuracies, ddof=1) / np.sqrt(len(accuracies)))
        results[n] = (average_accuracy, sem)
        print(f"n = {n}, Average Accuracy: {average_accuracy}, SEM: {sem}")

    ns = list(results.keys())
    mean_accuracies = [results[n][0] for n in ns]
    sems = [results[n][1] for n in ns]

    plt.figure(figsize=(1.7, 2.5))
    plt.errorbar([n + 1 for n in ns], mean_accuracies, yerr=sems, fmt="o-", capsize=5, color="b")
    plt.xlabel("Grid size")
    plt.ylabel("Average accuracy")
    plt.savefig("accuracy_by_grid_size.png", dpi=300, bbox_inches="tight")
    plt.show()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Script version of Graph sequence model.ipynb")
    parser.add_argument("--reuse-models", action="store_true", help="Skip training; use existing outputs_* folders.")
    parser.add_argument(
        "--n-walks-train",
        type=int,
        default=20,
        help="Number of random walks per graph/tree for training (default: 20).",
    )
    parser.add_argument(
        "--max-walk-length",
        type=int,
        default=50,
        help="Max length of each training walk (random in [1, N]; default: 20).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use much smaller datasets for a quick run (still executes the same pipeline).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for python/numpy randomness.")
    args = parser.parse_args(argv)

    os.chdir(INFERENCE_DIR)

    if args.seed is not None:
        import numpy as np

        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.smoke:
        n_graphs_train = 2000
        n_graphs_test = 50
        n_walks_test = 1
    else:
        n_graphs_train = 100000
        n_graphs_test = 100
        n_walks_test = 1

    n_walks_train = args.n_walks_train
    max_walk_length = args.max_walk_length

    family_model_path = "outputs_tree"
    spatial_model_path = "outputs_graph"

    if not args.reuse_models:
        _train_spatial(
            n_graphs_train=n_graphs_train,
            n_walks_train=n_walks_train,
            max_walk_length=max_walk_length,
            n_graphs_test=n_graphs_test,
            n_walks_test=n_walks_test,
        )
        _train_family(
            n_graphs_train=n_graphs_train,
            n_walks_train=n_walks_train,
            max_walk_length=max_walk_length,
            n_graphs_test=n_graphs_test,
            n_walks_test=n_walks_test,
        )

    loop_templates_spatial = [
        "{} EAST {} WEST {}",
        "{} WEST {} EAST {}",
        "{} NORTH {} SOUTH {}",
        "{} SOUTH {} NORTH {}",
        "{} EAST {} SOUTH {} WEST {} NORTH {}",
        "{} SOUTH {} WEST {} NORTH {} EAST {}",
        "{} WEST {} NORTH {} EAST {} SOUTH {}",
        "{} NORTH {} EAST {} SOUTH {} WEST {}",
        "{} EAST {} EAST {} NORTH {} WEST {} WEST {} SOUTH {}",
        "{} NORTH {} NORTH {} WEST {} SOUTH {} SOUTH {} EAST {}",
        # 8-hop: outer perimeter of 3×3 grid
        "{} EAST {} EAST {} SOUTH {} SOUTH {} WEST {} WEST {} NORTH {} NORTH {}",
        "{} SOUTH {} SOUTH {} WEST {} WEST {} NORTH {} NORTH {} EAST {} EAST {}",
    ]

    spatial_model_available = (
        Path(spatial_model_path, "model.safetensors").exists()
        or Path(spatial_model_path, "pytorch_model.bin").exists()
    )
    family_model_available = (
        Path(family_model_path, "model.safetensors").exists()
        or Path(family_model_path, "pytorch_model.bin").exists()
    )

    spatial_results_dict: dict[str, float] | None = None
    family_results_dict: dict[str, float] | None = None

    if spatial_model_available:
        model = GPT(base_model=spatial_model_path, base_model_name="gpt2")
        average_accuracy, spatial_results_dict = test_loop_greedy(model, loop_templates_spatial)
        print(f"Spatial Average Accuracy: {average_accuracy}")
    else:
        print(f"Spatial model weights not found in {spatial_model_path}/ – skipping spatial evaluation.")

    # 4-hop templates close with PARENT_OF or CHILD_OF (inverse relations)
    # so the model has a strong cue for the closing entity (like 2-hop).
    # Entity indices [0,1,2,1,0] mean: e1 e2 e3 e2 e1 (repeated middle for back-up steps).
    loop_templates_family: list[str | tuple[str, list[int]]] = [
        # 2-hop (inverse-pair templates)
        "{} CHILD_OF {} PARENT_OF {}",
        "{} PARENT_OF {} CHILD_OF {}",
        "{} GRANDCHILD_OF {} GRANDPARENT_OF {}",
        "{} GRANDPARENT_OF {} GRANDCHILD_OF {}",
        # 4-hop (close via PARENT_OF / CHILD_OF; middle entity repeats)
        ("{} CHILD_OF {} CHILD_OF {} PARENT_OF {} PARENT_OF {}", [0, 1, 2, 1, 0]),
        ("{} PARENT_OF {} PARENT_OF {} CHILD_OF {} CHILD_OF {}", [0, 1, 2, 1, 0]),
        ("{} CHILD_OF {} SPOUSE_OF {} SPOUSE_OF {} PARENT_OF {}", [0, 1, 2, 1, 0]),
        ("{} PARENT_OF {} SPOUSE_OF {} SPOUSE_OF {} CHILD_OF {}", [0, 1, 2, 1, 0]),
        # 6-hop (close via single-target symmetric relations)
        "{} CHILD_OF {} SPOUSE_OF {} CHILD_OF {} SPOUSE_OF {} GRANDPARENT_OF {} SIBLING_OF {}",
        "{} GRANDPARENT_OF {} SIBLING_OF {} CHILD_OF {} SPOUSE_OF {} CHILD_OF {} SPOUSE_OF {}",
    ]

    if family_model_available:
        model = GPT(base_model=family_model_path, base_model_name="gpt2")
        average_accuracy, family_results_dict = test_loop_greedy(model, loop_templates_family)
        print(f"Family Average Accuracy: {average_accuracy}")
    else:
        print(f"Family model weights not found in {family_model_path}/ – skipping family evaluation.")

    if spatial_results_dict is not None and family_results_dict is not None:
        _plot_aggregated_inf(family_results_dict=family_results_dict, spatial_results_dict=spatial_results_dict)
    elif family_results_dict is not None:
        _plot_aggregated_inf(family_results_dict=family_results_dict, spatial_results_dict={})
    elif spatial_results_dict is not None:
        _plot_aggregated_inf(family_results_dict={}, spatial_results_dict=spatial_results_dict)

    if spatial_model_available:
        _plot_loss(Path(spatial_model_path) / "trainer_state.json", "spatial_loss.png")
    if family_model_available:
        _plot_loss(Path(family_model_path) / "trainer_state.json", "family_loss.png")

    # Keep notebook's track_coordinates self-tests.
    walks1 = ["sz WEST zr EAST zr"]  # This should be invalid
    walks2 = ["ab EAST xy NORTH yz"]  # This should be valid
    print(track_coordinates(walks1))  # Expected output: False
    print(track_coordinates(walks2))  # Expected output: True

    if spatial_model_available:
        _imagination(spatial_model_path)
        _grid_generalisation(spatial_model_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
