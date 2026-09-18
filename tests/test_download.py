import pytest

from fabguard.download import load_downloads


def test_pinned_release_contains_sixty_unique_csv_files():
    downloads = load_downloads("configs/uored_v5_downloads.tsv")

    assert len(downloads) == 60
    assert len({name for name, _ in downloads}) == 60
    assert {name.split("_", 1)[0] for name, _ in downloads} == {"H", "I", "O", "B", "C"}


@pytest.mark.parametrize(
    "unsafe_name", ["../escape.csv", "nested/escape.csv", "nested\\escape.csv"]
)
def test_download_list_rejects_path_like_filenames(tmp_path, unsafe_name):
    rows = [
        f"sample-{index}.csv\thttps://data.mendeley.com/public-files/datasets/example/{index}"
        for index in range(59)
    ]
    rows.append(f"{unsafe_name}\thttps://data.mendeley.com/public-files/datasets/example/bad")
    manifest = tmp_path / "downloads.tsv"
    manifest.write_text("\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="invalid download row"):
        load_downloads(manifest)
