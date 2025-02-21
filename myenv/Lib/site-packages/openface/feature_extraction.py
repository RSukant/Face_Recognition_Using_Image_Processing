from typing import Union

from pathlib import Path
import time
import subprocess
import shlex
import shutil


BINARY_PATH = Path(__file__).parents[1] / "bin" / "FeatureExtraction"
BASE_OPTIONS = "-2Dfp -3Dfp -pdmparams -pose -aus -gaze -au_static"


def feature_extraction(
    video_path: Union[str, Path],
    features_path: Union[str, Path],
    verbose: Union[bool, int] = True,
    overwrite: bool = False,
):
    if not isinstance(video_path, Path):
        video_path = Path(video_path)
    if not isinstance(features_path, Path):
        features_path = Path(features_path)

    assert video_path.is_file()

    tmp_dir = Path(f"/tmp/TEMP_{time.time()}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    args = f"{BINARY_PATH} -f {video_path} -out_dir {tmp_dir} {BASE_OPTIONS}"

    if verbose:
        message = (
            "OpenFace will run with the following command:\n"
            f"{args}\n"
            "It might take a while, depending on the length of the video and the number of CPUs."
        )
        print(message)

    try:
        start = time.time()
        process = subprocess.run(
            args=shlex.split(args),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        end = time.time()
        if verbose > 1:
            print(process.stdout)
        if verbose:
            print(f"OpenFace took {end - start}.")
    except subprocess.CalledProcessError as error:
        print(error.stdout)

    if features_path.exists():
        if overwrite:
            features_path.unlink()
        else:
            raise FileExistsError(features_path)
    features_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = tmp_dir / (video_path.stem + ".csv")
    tmp_path.replace(features_path)

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    feature_extraction(
        "/home/guillaume/Datasets/UNIL/CH.101/Videos/CH.101.h.mp4",
        "/home/guillaume/Datasets/UNIL/CH.101/FacialAnalysis/CH.101.f.csv",
        verbose=2,
        overwrite=True,
    )
