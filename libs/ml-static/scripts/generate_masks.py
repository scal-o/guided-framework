from pathlib import Path

import geopandas as gpd
import torch


def generate_and_save_masks(city_name, seed=42):
    """Generate and save boolean masks for a given city."""
    torch.manual_seed(seed)

    data_dir = Path(f"networks/{city_name}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist. Please check the path.")

    links = gpd.read_file(data_dir / f"{city_name}_master.gpkg", layer="links")
    num_links = len(links)

    mask = torch.rand(num_links)
    mask_path = data_dir / "link_mask.pt"
    torch.save(mask, mask_path)


cities = ["sioux_falls", "anaheim", "chicago"]

if __name__ == "__main__":
    for city in cities:
        generate_and_save_masks(city)
