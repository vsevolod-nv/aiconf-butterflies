from __future__ import annotations

import json
from io import BytesIO
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
from PIL import Image, ImageFilter
from tqdm import tqdm


JsonDict = dict[str, Any]
FloatArray = npt.NDArray[np.float32]


class PhotoRow(TypedDict):
    photo_id: int
    photo_url: str
    observation_id: int | None
    observed_on: str | None
    author: str | int | None
    taxon_id: int | None
    taxon_name: str | None
    taxon_preferred_common_name: str | None


class ImageScores(TypedDict):
    contrast_score: float
    brightness_score: float
    edge_density: float
    center_contrast_score: float
    center_detail_score: float
    outer_detail_score: float
    detail_ratio: float
    active_blocks_ratio: float
    dominant_block_ratio: float


class HardPhotoRow(TypedDict):
    photo_id: int
    photo_url: str
    hard: bool


class HardPhotoDebugRow(HardPhotoRow):
    contrast_score: float
    brightness_score: float
    edge_density: float
    center_contrast_score: float
    center_detail_score: float
    outer_detail_score: float
    detail_ratio: float
    active_blocks_ratio: float
    dominant_block_ratio: float
    hard_score: int
    hard_signals: list[str]


class CollectedPhotoRow(PhotoRow):
    hard: bool


class CollectedPhotoDebugRow(CollectedPhotoRow):
    contrast_score: float
    brightness_score: float
    edge_density: float
    center_contrast_score: float
    center_detail_score: float
    outer_detail_score: float
    detail_ratio: float
    active_blocks_ratio: float
    dominant_block_ratio: float
    hard_score: int
    hard_signals: list[str]


class BasicPhotoRow(TypedDict):
    photo_id: int
    photo_url: str


def inat_get(endpoint: str, params: JsonDict | None = None) -> JsonDict:
    url = f"https://api.inaturalist.org/v1/{endpoint.lstrip('/')}"
    response = requests.get(url, params=params, timeout=30)
    return response.json()


def collect_inat_photos(
    taxon_id: int | None,
    target_photo_count: int,
    photo_license: str = "cc0",
    per_page: int = 200,
    unique_authors: bool = True,
    annotate_hard: bool = False,
    add_info: bool = True,
    observation_params: JsonDict | None = None,
    **hard_kwargs: Any,
) -> pd.DataFrame:
    rows: list[
        BasicPhotoRow
        | PhotoRow
        | HardPhotoRow
        | HardPhotoDebugRow
        | CollectedPhotoRow
        | CollectedPhotoDebugRow
    ] = []
    seen_photo_ids: set[int] = set()
    seen_authors: set[str | int | None] = set()
    page = 1
    debug = bool(hard_kwargs.pop("debug", False))
    progress_bar = tqdm(total=target_photo_count, desc="Collecting photos")

    while len(rows) < target_photo_count:
        params: JsonDict = {
            "photos": "true",
            "photo_license": photo_license,
            "per_page": per_page,
            "page": page,
        }
        if taxon_id is not None:
            params["taxon_id"] = taxon_id
        if observation_params:
            params.update(observation_params)

        payload = inat_get(
            "observations",
            params,
        )
        observations = payload.get("results", [])
        progress_bar.set_postfix(page=page, collected=len(rows))
        if not observations:
            break

        collected_before_page = len(rows)
        for observation in observations:
            user = observation.get("user") or {}
            author = user.get("login") or user.get("id")
            taxon = observation.get("taxon") or {}

            if unique_authors and author in seen_authors:
                continue

            for photo in observation.get("photos", []):
                photo_id = photo.get("id")
                photo_url = photo.get("original_url") or photo.get("url", "").replace("square", "original")

                if photo_id in seen_photo_ids:
                    continue
                if photo.get("license_code") != photo_license:
                    continue
                if not photo_url:
                    continue

                seen_photo_ids.add(photo_id)
                if unique_authors:
                    seen_authors.add(author)

                base_row = PhotoRow(
                    photo_id=photo_id,
                    photo_url=photo_url,
                    observation_id=observation.get("id"),
                    observed_on=observation.get("observed_on"),
                    author=author,
                    taxon_id=taxon.get("id"),
                    taxon_name=taxon.get("name"),
                    taxon_preferred_common_name=taxon.get("preferred_common_name"),
                )
                if annotate_hard:
                    scores, hard_score, hard_signals, hard = classify_hard_photo(photo_url, **hard_kwargs)
                    if add_info:
                        row: PhotoRow | HardPhotoRow | HardPhotoDebugRow | CollectedPhotoRow | CollectedPhotoDebugRow = CollectedPhotoRow(
                            **base_row,
                            hard=hard,
                        )
                    else:
                        row = HardPhotoRow(
                            photo_id=photo_id,
                            photo_url=photo_url,
                            hard=hard,
                        )
                    if debug:
                        if add_info:
                            row = CollectedPhotoDebugRow(
                                **row,
                                **scores,
                                hard_score=hard_score,
                                hard_signals=hard_signals,
                            )
                        else:
                            row = HardPhotoDebugRow(
                                **row,
                                **scores,
                                hard_score=hard_score,
                                hard_signals=hard_signals,
                            )
                else:
                    row = base_row if add_info else BasicPhotoRow(
                        photo_id=photo_id,
                        photo_url=photo_url,
                    )

                rows.append(row)
                progress_bar.update(1)
                break

            if len(rows) >= target_photo_count:
                break

        progress_bar.set_postfix(
            page=page,
            collected=len(rows),
            added=len(rows) - collected_before_page,
        )
        progress_bar.refresh()
        page += 1

    progress_bar.close()
    return pd.DataFrame(rows)


def load_image(url: str, image_size: int = 512) -> Image.Image:
    image = Image.open(BytesIO(requests.get(url, timeout=30).content)).convert("L")
    image.thumbnail((image_size, image_size))
    return image


def split_center_and_outer(
    image_array: FloatArray, center_fraction: float = 0.45
) -> tuple[FloatArray, FloatArray]:
    height, width = image_array.shape
    crop_height = max(1, int(height * center_fraction))
    crop_width = max(1, int(width * center_fraction))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    center = image_array[top : top + crop_height, left : left + crop_width]
    outer_mask = np.ones_like(image_array, dtype=bool)
    outer_mask[top : top + crop_height, left : left + crop_width] = False
    outer = image_array[outer_mask]
    return center, outer


def gaussian_detail_map(image: Image.Image, radius: int = 6) -> tuple[FloatArray, FloatArray]:
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    image_array = np.asarray(image, dtype=np.float32)
    blurred_array = np.asarray(blurred, dtype=np.float32)
    return image_array, np.abs(image_array - blurred_array)


def block_scores(detail_map: FloatArray, grid_size: int = 6) -> tuple[float, float]:
    rows = np.array_split(detail_map, grid_size, axis=0)
    means: list[float] = []
    for row in rows:
        for block in np.array_split(row, grid_size, axis=1):
            means.append(float(block.mean()))
    block_means = np.array(means, dtype=np.float32)
    active_blocks_ratio = float((block_means >= block_means.mean()).mean())
    dominant_block_ratio = float(block_means.max() / (block_means.mean() + 1e-6))
    return active_blocks_ratio, dominant_block_ratio


def image_scores(
    image: Image.Image, center_fraction: float = 0.45, gaussian_radius: int = 6
) -> ImageScores:
    image_array, detail_map = gaussian_detail_map(image, radius=gaussian_radius)
    center, _ = split_center_and_outer(image_array, center_fraction=center_fraction)
    center_detail, outer_detail = split_center_and_outer(detail_map, center_fraction=center_fraction)
    edge_x = np.abs(np.diff(image_array, axis=1))
    edge_y = np.abs(np.diff(image_array, axis=0))
    active_blocks_ratio, dominant_block_ratio = block_scores(detail_map)
    return ImageScores(
        contrast_score=float(image_array.std()),
        brightness_score=float(image_array.mean()),
        edge_density=float(((edge_x > 12).mean() + (edge_y > 12).mean()) / 2),
        center_contrast_score=float(center.std()),
        center_detail_score=float(center_detail.mean()),
        outer_detail_score=float(outer_detail.mean()),
        detail_ratio=float(center_detail.mean() / (outer_detail.mean() + 1e-6)),
        active_blocks_ratio=active_blocks_ratio,
        dominant_block_ratio=dominant_block_ratio,
    )


def classify_hard_photo(
    photo_url: str,
    image_size: int = 512,
    contrast_max: float = 56,
    brightness_min: float = 60,
    brightness_max: float = 190,
    edge_density_min: float = 0.22,
    detail_ratio_max: float = 1.03,
    active_blocks_ratio_min: float = 0.50,
    dominant_block_ratio_max: float = 1.65,
    hard_threshold: int = 4,
) -> tuple[ImageScores, int, list[str], bool]:
    scores = image_scores(load_image(photo_url, image_size=image_size))
    signals: dict[str, bool] = {
        "low_contrast": scores["contrast_score"] <= contrast_max,
        "too_dark": scores["brightness_score"] <= brightness_min,
        "too_bright": scores["brightness_score"] >= brightness_max,
        "busy_background": scores["edge_density"] >= edge_density_min,
        "diffuse_detail": scores["active_blocks_ratio"] >= active_blocks_ratio_min,
        "no_dominant_subject": scores["dominant_block_ratio"] <= dominant_block_ratio_max,
        "low_local_focus_gain": scores["detail_ratio"] <= detail_ratio_max,
    }
    hard_score = int(sum(signals.values()))
    return scores, hard_score, [name for name, active in signals.items() if active], hard_score >= hard_threshold


def annotate_hard_photos(df: pd.DataFrame, debug: bool = False, **kwargs: Any) -> pd.DataFrame:
    rows: list[HardPhotoRow | HardPhotoDebugRow] = []

    for row in tqdm(df.to_dict("records"), total=len(df), desc="Scoring photos"):
        scores, hard_score, hard_signals, hard = classify_hard_photo(row["photo_url"], **kwargs)
        result: HardPhotoRow | HardPhotoDebugRow = HardPhotoRow(
            photo_id=row["photo_id"],
            photo_url=row["photo_url"],
            hard=hard,
        )
        if debug:
            result = HardPhotoDebugRow(
                **result,
                **scores,
                hard_score=hard_score,
                hard_signals=hard_signals,
            )
        rows.append(result)

    return pd.DataFrame(rows)


def save_df_json(df: pd.DataFrame, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(df.to_dict("records"), file, indent=2, ensure_ascii=False)
