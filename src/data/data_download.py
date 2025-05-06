import os
import zipfile

import click
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()


@click.command()
@click.option("--kaggle-username", envvar="KAGGLE_USERNAME", help="Kaggle username for authentication")
@click.option("--kaggle-key", envvar="KAGGLE_KEY", help="Kaggle key for authentication")
@click.option(
    "--competition",
    "-c",
    envvar="KAGGLE_COMPETITION",
    default="asr-numbers-recognition-in-russian",
    show_envvar=True,
    help="Name of the Kaggle competition to download",
)
@click.option(
    "--output-dir",
    "-o",
    envvar="INITIAL_DATA_PATH",
    default="data",
    show_envvar=True,
    help="Directory to save and extract competition data",
)
def main(kaggle_username: str, kaggle_key: str, competition: str, output_dir: str):
    """
    Download and extract Kaggle competition data.
    """
    if os.environ.get("KAGGLE_USERNAME") != kaggle_username or os.environ.get("KAGGLE_KEY") != kaggle_key:
        os.environ["KAGGLE_USERNAME"] = kaggle_username
        os.environ["KAGGLE_KEY"] = kaggle_key

    api = KaggleApi()
    api.authenticate()

    os.makedirs(output_dir, exist_ok=True)

    archive_path = os.path.join(output_dir, f"{competition}.zip")
    click.echo(f"Downloading '{competition}' to '{archive_path}'...")
    api.competition_download_files(competition=competition, path=output_dir)

    # Extract files
    click.echo(f"Extracting '{archive_path}' into '{output_dir}'...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(output_dir)

    click.echo(f"Data ready in: {output_dir}")


if __name__ == "__main__":
    main()
