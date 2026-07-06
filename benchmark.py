import gc
import os
import csv
import torch
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path

from nav.pointnav_agent import PointnavAgent
from utils.frontier_utils import read_config_yaml
from utils.config_utils import ovon_config, hm3d_config, mp3d_config, DATA_PATH

import habitat


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def write_metrics(metrics, path="objnav_hm3d.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_episodes", type=int, default=-1)

    p.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="hm3d, mp3d, or OVON",
    )


    p.add_argument(
        "--nickname",
        type=str,
        required=True,
        help="Nickname for the benchmark run",
    )

    p.add_argument(
        "--config",
        type=str,
        default="config/hm3d_navigation.yaml",
        help="OpenFrontier configuration file",
    )

    p.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum number of navigation steps",
    )
    p.add_argument(
        "--max_time", type=int, default=3600, help="Maximum navigation time in seconds"
    )
    p.add_argument(
        "--vis_graph",
        action="store_true",
        default=False,
        help="Visualize the topological graph",
    )
    p.add_argument(
        "--save-images",
        "--save_images",
        dest="save_images",
        action="store_true",
        default=False,
        help="Save intermediate navigation images during benchmark episodes",
    )
    p.add_argument(
        "--unet_weight",
        type=Path,
        default=Path("model_weights/rgbd_11cls.pth"),
        help="Path to UNet model weights",
    )

    p.add_argument(
        "--log_level",
        "-ll",
        type=int,
        default=20,
        help="logging level (0=notset, 10=debug, 20=info...)",
    )

    p.add_argument(
        "--split",
        type=int,
        default=(1, 1),
        nargs=2,
        help="Split number for evaluation in the format #subset_index #total_subsets",
    )
    
    p.add_argument(
        "--output-path",
        type=str,
        required=True,
    )

    return p.parse_known_args()[0]


if __name__ == "__main__":
    args = get_args()

    # list all the scenes in objnav path

    benchmark = args.benchmark.lower()
    output_path = args.output_path.lower()

    if benchmark == "ovon":
        directory = Path(DATA_PATH + "ovon/val_unseen/content/")
    elif benchmark == "hm3d":
        directory = Path(DATA_PATH + f"objectnav_hm3d_v2/val/content/")
    elif benchmark == "mp3d":
        directory = Path(DATA_PATH + f"objectnav_mp3d_v1/val/content/")
    else:
        raise ValueError(f"Benchmark {benchmark} not recognized")

    scenes = sorted([f.stem.split(".")[0] for f in directory.glob("*.json.gz")])

    if len(scenes) == 0:
        raise ValueError(f"No scenes found in {directory}")

    # split the scenes based on args.split and select the corresponding subset
    total_splits = args.split[1]
    split_index = args.split[0] - 1
    scenes = [
        scene for i, scene in enumerate(scenes) if i % total_splits == split_index
    ]

    scenes_data = {}
    killed = False
    exception = None

    fn_config = read_config_yaml(args.config)

    probabilities_source = (
        fn_config.get("probabilities_source", "gemini-2.5").lower().split("-")[0]
    )
    detection_source = (
        fn_config.get("detection_source", "gemini-2.5").lower().split("-")[0]
    )
    segmentation_source = (
        fn_config.get("segmentation_source", "sam3").lower().split("-")[0]
    )

    benchmark_nickname = args.nickname

    output_dir = Path(
        f"{output_path}/{benchmark_nickname}_{segmentation_source}_{probabilities_source}_{detection_source}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        if killed:
            raise exception if exception is not None else KeyboardInterrupt()
        
        killed = False
        exception = None

        print("Evaluating Scene: %s" % scene)

        if args.benchmark.lower() == "hm3d":
            habitat_config = hm3d_config(stage=scene, episodes=args.eval_episodes)
        elif args.benchmark.lower() == "mp3d":
            habitat_config = mp3d_config(stage=scene, episodes=args.eval_episodes)
        elif args.benchmark.lower() == "ovon":
            habitat_config = ovon_config(stage=scene, episodes=args.eval_episodes)
        else:
            raise ValueError("Benchmark not recognized: %s" % args.benchmark)

        try:
            habitat_env = habitat.Env(habitat_config)
        except Exception as e:
            print(e)
            continue

        if args.eval_episodes == -1:
            num_episodes = habitat_env.number_of_episodes
        else:
            num_episodes = args.eval_episodes

        # Inside output dir
        scene_dir = Path.joinpath(output_dir, scene)
        scene_dir.mkdir(parents=True, exist_ok=True)

        metrics_dir = Path.joinpath(output_dir, "metrics/%s.csv" % scene)
        metrics_dir.parent.mkdir(parents=True, exist_ok=True)

        evaluation_metrics = []

        # Load existing metrics
        if metrics_dir.exists():
            with open(metrics_dir, mode="r") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    evaluation_metrics.append(
                        {
                            "episode": int(row["episode"]),
                            "success": float(row["success"]),
                            "spl": float(row["spl"]),
                            "distance_to_goal": float(row["distance_to_goal"]),
                            "object_goal": row["object_goal"],
                            "termination_reason": row["termination_reason"],
                        }
                    )

        # check how many episodes have been evaluated for this scene
        evaluated_episodes = len(evaluation_metrics)
        
        if evaluated_episodes >= num_episodes:
            print(
                "All %d episodes for scene %s have already been evaluated. Skipping..."
                % (num_episodes, scene)
            )
            habitat_env.close()
            continue


        for i in tqdm(range(num_episodes)):
            if killed:
                break

            habitat_env.reset()

            target = habitat_env.current_episode.object_category
            
            target_name = target.replace(" ", "_")
            folder_name = f"episode-{i}-{target_name}"

            path = scene_dir / folder_name
            make_dir = Path(path)

            if any(m["episode"] == i for m in evaluation_metrics):
                print(
                    "Episode %d for scene %s has already been evaluated. Skipping..."
                    % (i, scene)
                )
                continue

            episode = habitat_env.current_episode

            habitat_agent = PointnavAgent(
                habitat_env,
                args,
                save_dir=path,
                openfrontier_config=fn_config,
                habitat_config=habitat_config,
                scene=scene,
            )
 
            habitat_agent.setup_system()
            habitat_agent.initialize()


            reason = "unknown"
            try:
                killed = False
                exception = None
                while (
                    not habitat_env.episode_over
                    and habitat_agent.navigation_steps <= 500
                ):

                    navigate, reason = habitat_agent.navigation(
                        save_images=args.save_images
                    )
                            
                    habitat_agent.update_video()

                    if not navigate:
                        habitat_env.step(0)

            except Exception as e:
                exception = e
                # If keyboard interrupt, stop evaluation
                if isinstance(e, KeyboardInterrupt):
                    killed = True
                    print("Keyboard Interrupt. Stopping evaluation...")
                    reason = "keyboard_interrupt"
                elif "exhausted" in str(e).lower():
                    killed = True
                    print("All API keys exhausted. Stopping evaluation...")
                    reason = "api_keys_exhausted"
                elif "banned" in str(e).lower():
                    killed = True
                    print("Account banned. Stopping evaluation...")
                    reason = "account_banned"
                else:
                    print("Exception during evaluation of episode %d: %s" % (i, str(e)))
                    reason = "exception_occurred"

                    with open(path / "exception.txt", "w") as f:
                        # Write the full exception traceback to the file
                        import traceback

                        traceback.print_exc(file=f)

            finally:
                try:
                    metrics = habitat_env.get_metrics()

                    if int(metrics["success"]) == 0 and reason == "object_found":
                        reason = "false_positive"

                    episode_metrics = {
                        "episode": i,
                        "success": metrics["success"],
                        "spl": metrics["spl"],
                        "distance_to_goal": metrics["distance_to_goal"],
                        "object_goal": episode.object_category,
                        "termination_reason": reason,
                    }

                    episode_metrics_dir = Path.joinpath(path, "metrics.csv")
                    write_metrics(
                        [episode_metrics],
                        episode_metrics_dir,
                    )

                    habitat_agent.save_trajectory(path)

                    # subsitute if already existing
                    evaluation_metrics = [
                        m for m in evaluation_metrics if m["episode"] != i
                    ]
                    evaluation_metrics.append(episode_metrics)
                    write_metrics(evaluation_metrics, path=metrics_dir)

                    if metrics["success"]:
                        # move to success folder
                        success_path = scene_dir / f"success"

                        success_path.mkdir(parents=True, exist_ok=True)

                        new_folder = success_path / folder_name
                        # if folder exists, remove
                        if new_folder.exists():
                            shutil.rmtree(new_folder)

                        os.rename(path, new_folder)
                    else:
                        # move to failure folder
                        failure_path = scene_dir / f"failure"

                        failure_path.mkdir(parents=True, exist_ok=True)

                        folder_name = folder_name + f"-{reason}/"
                        new_folder = failure_path / folder_name

                        if new_folder.exists():
                            shutil.rmtree(new_folder)

                        os.rename(path, new_folder)
                    
                    del habitat_agent
                    del episode
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass

                except Exception as e:
                    print("Exception during metrics logging: %s" % str(e))

        habitat_env.close()
        del habitat_env
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            pass

        print("Closed environment for scene: %s" % scene)
