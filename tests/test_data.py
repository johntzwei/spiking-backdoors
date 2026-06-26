"""Unit tests for the MIA data splitting (no GPU or network needed)."""

from hubble.data import split_items


def make_records(n=100):
    # Half non-members (label 0) and half members at dup=16, with stable ids.
    records = []
    for i in range(n):
        is_member = i % 2 == 0
        records.append(
            {
                "id": i,
                "text": f"passage {i}",
                "duplicates": 16 if is_member else 0,
                "label": 1 if is_member else 0,
            }
        )
    return records


def test_split_is_deterministic_across_loads():
    # split_items sorts by id and fixes the seed, so two independent calls must produce
    # the exact same ordering of examples in both halves.
    records = make_records()

    first_train, first_test = split_items(records, dup=16)
    second_train, second_test = split_items(records, dup=16)

    first_train_ids = [record["id"] for record in first_train]
    second_train_ids = [record["id"] for record in second_train]
    first_test_ids = [record["id"] for record in first_test]
    second_test_ids = [record["id"] for record in second_test]

    assert first_train_ids == second_train_ids
    assert first_test_ids == second_test_ids
