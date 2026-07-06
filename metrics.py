import csv
from pathlib import Path
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--paths",
    nargs="+",
    type=str,
    required=True,
    help="Paths to the directories containing /metrics folders",
)
args = parser.parse_args()

paths = args.paths

full_csv = []
summary = []
all_success_rate = 0
all_avg_spl = 0
all_count = 0
all_objects = {}
all_reasons = {}
all_false_positives = {}
all_max_steps = {}
total_count = 0

for path in paths:

    path += "/metrics"
    parent = Path(path).parent

    def get_image(path):
        Path(str(parent) + path).mkdir(exist_ok=True)

        img_path = f"scene_{scene}_episode_{row[1]}.png"

        get_image = True
        folder = None
        for dirpath, _, filenames in os.walk(str(parent) + path):
            if img_path in filenames:
                img_path = os.path.join(dirpath, img_path)
                folder = os.path.dirname(img_path).split("/")[-1]
                get_image = False
                break

        if get_image:
            # Get the latest frame in the metrics.mp4 file
            mp4_path = str(parent) + f"/{scene}/episode-{row[1]}/metrics.mp4"
            mp4_path = Path(mp4_path)
            if mp4_path.exists():
                cap = cv2.VideoCapture(str(mp4_path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                ret, frame = cap.read()
                if ret:
                    cv2.putText(
                        frame,
                        f"Goal: {object_goal}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    # get the image inside termination_visualizations folder and copy it to max_steps_videos folder additionally
                    term_viz_path = (
                        str(parent)
                        + f"/{scene}/episode-{row[1]}/termination_visualizations"
                    )
                    term_viz_path = Path(term_viz_path)
                    images = []
                    if term_viz_path.exists():
                        # Get all png files in the termination_visualizations folder
                        for file in term_viz_path.glob("*.png"):
                            images.append(file)

                    # Join all images to the frame
                    for img_file in images:
                        img = cv2.imread(str(img_file))
                        # Resize img to have the same height as frame
                        height, width, _ = frame.shape
                        img_height, img_width, _ = img.shape
                        scale = height / img_height
                        new_width = int(img_width * scale)
                        resized_img = cv2.resize(img, (new_width, height))
                        # Concatenate images horizontally
                        frame = cv2.hconcat([frame, resized_img])

                    # Save the frame as an image
                    cv2.imwrite(
                        str(parent) + f"/{path}/" + img_path,
                        frame,
                    )
                cap.release()

        return folder

    # Open all CSV files in the metrics directory

    for scene_idx, csv_file in enumerate(Path(path).glob("*.csv")):
        with open(csv_file, mode="r") as f:
            scene = csv_file.stem
            if scene[0] == "_":
                continue

            reader = csv.reader(f)
            # Skip header
            next(reader)
            success_rate = 0
            avg_spl = 0
            count = 0
            objects = {}
            reasons = {}
            false_positives = {}
            max_steps = {}
            for row in reader:
                row.insert(0, scene)  # Add scene as the first column

                if row[4] == "inf":  # or row[6] in ["api_keys_exhausted", "unknown"]:
                    continue

                full_csv.append(row)
                success_rate += float(row[2])
                all_success_rate += float(row[2])
                avg_spl += float(row[3])
                all_avg_spl += float(row[3])
                count += 1
                all_count += 1
                object_goal = row[5]
                if object_goal not in objects:
                    objects[object_goal] = {"success": 0, "count": 0}
                objects[object_goal]["count"] += 1
                objects[object_goal]["success"] += float(row[2])

                success = float(row[2])
                reason = row[6]

                if reason == "continue_navigation":
                    reason = "max_steps_reached"

                episode = row[1]

                total_count += 1

                if success < 1:
                    if reason not in reasons:
                        reasons[reason] = 0

                    reasons[reason] += 1

                    exception_path = (
                        str(parent)
                        + f"/{scene}/failure/episode-{row[1]}-{reason}/exception.txt"
                    )
                    if Path(exception_path).exists():
                        reason = "exception"
                        with open(exception_path, mode="r") as ex_file:
                            ex_lines = ex_file.readlines()
                            print(ex_lines)
                    else:
                        reason = "unknown"

            success_rate /= count
            avg_spl /= count
            for obj, vals in objects.items():
                obj_success_rate = vals["success"] / vals["count"]
                if obj not in all_objects:
                    all_objects[obj] = {"success": 0, "count": 0}
                all_objects[obj]["success"] += vals["success"]
                all_objects[obj]["count"] += vals["count"]

            for reason, val in reasons.items():
                if reason not in all_reasons:
                    all_reasons[reason] = 0
                all_reasons[reason] += val

            for folder, val in false_positives.items():
                if folder not in all_false_positives:
                    all_false_positives[folder] = 0
                all_false_positives[folder] += val

            for reason, val in max_steps.items():
                if reason not in all_max_steps:
                    all_max_steps[reason] = 0
                all_max_steps[reason] += val

            # all_success_rate += success_rate
            # all_avg_spl += avg_spl
            # all_count += 1

            print(f"{scene_idx}\t{scene}\t{count}\t{success_rate:.4f}\t{avg_spl:.4f}")
            summary.append([scene, f"{success_rate:.4f}", f"{avg_spl:.4f}"])


# Print per-object success rates
object_success_rates = []
for obj, vals in all_objects.items():
    obj_success_rate = vals["success"] / vals["count"]
    object_success_rates.append((obj, obj_success_rate))
    print(f"Object: {obj}, Success Rate: {obj_success_rate:.4f}")

# Print failure reasons
reason_rates = []
for reason, val in all_reasons.items():
    reason_rates.append((reason, val))

max_steps_rates = []
for reason, val in all_max_steps.items():
    max_steps_rates.append((reason, val))

# Write to a single CSV file

with open(str(parent) + "/full_metrics.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(
        ["scene", "episode", "success", "spl", "distance_to_goal", "object_goal"]
    )
    # Write all rows
    writer.writerows(full_csv)

# Write summary CSV file
with open(str(parent) + "/summary_metrics.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(["scene", "success_rate", "average_spl"])
    # Write summary rows
    writer.writerows(summary)

    writer.writerow([])
    writer.writerow(["object_goal", "success_rate"])
    for obj, rate in sorted(object_success_rates):
        writer.writerow([obj, f"{rate:.4f}"])

    writer.writerow([])
    writer.writerow(
        [
            "Overall",
            f"{(all_success_rate / all_count):.4f}",
            f"{(all_avg_spl / all_count):.4f}",
        ]
    )

    print(
        "Overall",
        f"{int(all_success_rate)}/{all_count}",
        f"{(all_success_rate / all_count):.4f}",
        f"{(all_avg_spl / all_count):.4f}",
    )

    writer.writerow([])
    writer.writerow(["failure_reason", "count"])
    for reason, count in sorted(reason_rates):
        writer.writerow([reason, count])
        print(f"{reason}: {count}")

    writer.writerow([])
    writer.writerow(["false_positive_folder", "count"])
    for folder, count in sorted(all_false_positives.items()):
        writer.writerow([folder, count])

    writer.writerow([])
    writer.writerow(["max_steps_object_goal", "count"])
    for reason, count in sorted(max_steps_rates):
        writer.writerow([reason, count])


import plotly.graph_objects as go

success_rate = all_success_rate / all_count
failure_rate = 1 - success_rate

run_success = round(success_rate * total_count)
run_failure = round(failure_rate * total_count)

# ----- Build nodes -----
nodes = []
index = {}


def add_node(name):
    if name not in index:
        index[name] = len(nodes)
        nodes.append(name)


# Level -1
add_node("Run")

# Level 0
add_node("Success")
add_node("Failure")

# Level 1
for reason in all_reasons:
    add_node(reason)

# Level 2
for folder in all_false_positives:
    add_node(folder)

labels = [""] * len(nodes)

# Root
labels[index["Run"]] = f"Run ({int(total_count)})"

labels[index["Success"]] = f"Success ({success_rate * 100:.0f}%)"
labels[index["Failure"]] = f"Failure ({failure_rate * 100:.0f}%)"

for reason, count in all_reasons.items():
    percentage = count / sum(all_reasons.values())
    labels[index[reason]] = f"{reason} ({percentage * 100:.0f}%)"

for folder, count in all_false_positives.items():
    percentage = count / sum(all_false_positives.values())
    labels[index[folder]] = f"{folder} ({percentage * 100:.0f}%)"

# ----- Node positions -----
# Plotly expects lists matching len(nodes)
x = [0] * len(nodes)
y = [0] * len(nodes)

# Root
x[index["Run"]] = 0.0
y[index["Run"]] = 0.0  # high

# Success / Failure
x[index["Success"]] = 0.3
y[index["Success"]] = 0.6  # same height → straight flow

x[index["Failure"]] = 0.3
y[index["Failure"]] = 0.1  # lower → downward bending flow

# Failure reasons (stacked)
reason_list = list(all_reasons.keys())
for i, reason in enumerate(reason_list):
    x[index[reason]] = 0.6
    y[index[reason]] = 0.2 + i * 0.2

# False-positive folders (lower)
folder_list = list(all_false_positives.keys())
for i, folder in enumerate(folder_list):
    x[index[folder]] = 0.85
    y[index[folder]] = 0.2 + i * 0.1

# ----- Links -----
source = []
target = []
value = []

# Run → Success/Failure
source += [index["Run"], index["Run"]]
target += [index["Success"], index["Failure"]]
value += [run_success, run_failure]

# Failure → reasons
total_failure_reasons = sum(all_reasons.values())
for reason, count in all_reasons.items():
    flow = round((count / total_failure_reasons) * run_failure)
    source.append(index["Failure"])
    target.append(index[reason])
    value.append(flow)

# false positive reason → folders
if "false positive" in all_reasons and len(all_false_positives) > 0:
    fp_reason = "false positive"
    fp_reason_count = all_reasons[fp_reason]
    fp_reason_flow = (fp_reason_count / total_failure_reasons) * run_failure

    total_fp_folders = sum(all_false_positives.values())
    for folder, folder_count in all_false_positives.items():
        subflow = round(fp_reason_flow * (folder_count / total_fp_folders))
        source.append(index[fp_reason])
        target.append(index[folder])
        value.append(subflow)

# ----- Plot -----
fig = go.Figure(
    data=[
        go.Sankey(
            node=dict(
                pad=15, thickness=20, line=dict(width=0.5), label=labels, x=x, y=y
            ),
            link=dict(source=source, target=target, value=value),
        )
    ]
)

fig.update_layout(title_text="Navigation Success / Failure Breakdown", font_size=14)
fig.write_html(str(parent) + "/sankey_diagram.html")
