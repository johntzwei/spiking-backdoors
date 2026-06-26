"""Loading and splitting the Hubble Wikipedia-passages MIA data.

The `allegrolab/passages_wikipedia` dataset is laid out so that membership is encoded by
the dataset's *own* split:
  - the `train` split holds passages that WERE inserted into the perturbed model's training
    data (members), each tagged with a `duplicates` count in {1, 4, 16, 64, 256};
  - the `test` split holds passages that were NEVER inserted (non-members), all `duplicates=0`.
"""

from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET_ID = "allegrolab/passages_wikipedia"


def load_wikipedia_passages():
    """Return one flat list of records: {id, text, duplicates, label}.

    label = 1 (member) if the passage was inserted (duplicates > 0, the dataset's train
    split), else 0 (non-member, the dataset's dup=0 test split).
    """
    dataset = load_dataset(DATASET_ID)

    records = []
    for split in ("train", "test"):
        for row in dataset[split]:
            duplicates = row["duplicates"]
            records.append(
                {
                    # id is a stable, content-independent key so split_items can sort by it
                    # and produce the same partition on every run, regardless of load order.
                    "id": len(records),
                    "text": row["text"],
                    "duplicates": duplicates,
                    "label": 1 if duplicates > 0 else 0,
                }
            )
    return records


def split_items(records, dup, test_size=0.5, seed=42):
    """Pool non-members with members at one duplication level, then split over ITEMS.

    Returns (train_items, test_items) for the binary MIA task "dup=0 vs dup=`dup`".

    NOTE: [thought process] The split must be over items (passages), not over the dataset's
    own train/test split. That built-in split *is* the membership label (train=member,
    test=non-member), so reusing it would put every positive in train and every negative in
    test — the classifier would never see both classes. Instead we pool the two classes and
    carve out a fresh held-out set, so the supervised attack is judged on unseen passages.

    NOTE: [thought process] We sort by `id` before splitting and fix the seed so the function
    is deterministic: the same records always yield the same partition. This lets the cheap
    classifier step be re-run without disturbing which passages are held out.
    """
    non_members = [record for record in records if record["label"] == 0]
    members = [record for record in records if record["duplicates"] == dup]
    pooled = sorted(non_members + members, key=lambda record: record["id"])

    labels = [record["label"] for record in pooled]
    train_items, test_items = train_test_split(
        pooled,
        test_size=test_size,
        random_state=seed,
        stratify=labels,  # keep the member/non-member ratio identical in both halves
    )
    return train_items, test_items
