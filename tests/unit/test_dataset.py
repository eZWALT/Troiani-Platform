from troiani_platform.training.dataset import fingerprint_dataset, write_dataset_manifest


def test_manifest_hash_not_full_data(tmp_path):
    fp = write_dataset_manifest(tmp_path / "dataset", "demo", "1", ["shard-000", "shard-001"])
    again = fingerprint_dataset(tmp_path / "dataset")
    assert again is not None
    assert again.dataset_hash == fp.dataset_hash
    assert again.revision == fp.revision
    assert again.shards == ("shard-000", "shard-001")
