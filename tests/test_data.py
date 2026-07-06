"""Unit tests for the MIA data splitting (no GPU or network needed)."""

from collections import Counter

from hubble.data import attack_split, zero_vs_dup


def make_records(n=120):
    # Non-members (dup=0) plus members spread over several duplication levels, with stable ids.
    dup_cycle = [0, 0, 1, 4, 16, 64]
    records = []
    for i in range(n):
        duplicates = dup_cycle[i % len(dup_cycle)]
        records.append(
            {
                "id": i,
                "text": f"passage {i}",
                "duplicates": duplicates,
                "label": 1 if duplicates > 0 else 0,
            }
        )
    return records


def test_split_is_deterministic_across_loads():
    # attack_split sorts by id and fixes the seed, so two independent calls must produce the exact
    # same ordering of examples in both halves.
    records = make_records()

    first_train, first_test = attack_split(records)
    second_train, second_test = attack_split(records)

    assert [r["id"] for r in first_train] == [r["id"] for r in second_train]
    assert [r["id"] for r in first_test] == [r["id"] for r in second_test]


def test_split_covers_every_record_once():
    # One global split: every record lands in exactly one half, nothing dropped or duplicated.
    records = make_records()
    train_items, test_items = attack_split(records)

    all_ids = sorted(r["id"] for r in train_items + test_items)
    assert all_ids == [r["id"] for r in records]
    assert not ({r["id"] for r in train_items} & {r["id"] for r in test_items})


def test_split_is_stratified_by_dup_level():
    # Stratifying by duplication level keeps each level's share of the test half ~= test_size, so
    # both members (at every dup) and non-members are represented on both sides of the split.
    records = make_records(n=600)
    train_items, test_items = attack_split(records, test_size=0.5)

    train_dups = Counter(r["duplicates"] for r in train_items)
    test_dups = Counter(r["duplicates"] for r in test_items)
    for dup in {r["duplicates"] for r in records}:
        assert abs(train_dups[dup] - test_dups[dup]) <= 1


def test_zero_vs_dup_selects_only_that_level_and_non_members():
    # The 0-vs-k eval set is exactly the non-members plus the members at that one dup level; other
    # dup levels are excluded, and labels line up with the returned items.
    records = make_records(n=600)
    n_non_members = sum(1 - r["label"] for r in records)
    n_dup16 = sum(r["duplicates"] == 16 for r in records)

    eval_items, labels = zero_vs_dup(records, dup=16)

    assert sum(labels) == n_dup16                              # positives = members at dup=16
    assert len(labels) - sum(labels) == n_non_members         # negatives = all dup=0 non-members
    assert labels == [item["label"] for item in eval_items]   # labels align with items
    assert {item["duplicates"] for item in eval_items} == {0, 16}
