from __future__ import annotations

"""HumanActivity 原始文本解析与缓存。"""

import hashlib
from pathlib import Path

import torch

from .cache import get_records_cache_path


class HumanActivityRawDataset:
    """把原始文本解析成记录级张量。"""

    tag_ids = [
        "010-000-024-033",
        "010-000-030-096",
        "020-000-033-111",
        "020-000-032-221",
    ]
    tag_dict = {key: index for index, key in enumerate(tag_ids)}

    label_names = [
        "walking",
        "falling",
        "lying down",
        "lying",
        "sitting down",
        "sitting",
        "standing up from lying",
        "on all fours",
        "sitting on the ground",
        "standing up from sitting",
        "standing up from sit on grnd",
    ]

    label_dict = {
        "walking": 0,
        "falling": 1,
        "lying": 2,
        "lying down": 2,
        "sitting": 3,
        "sitting down": 3,
        "standing up from lying": 4,
        "standing up from sitting": 4,
        "standing up from sit on grnd": 4,
        "on all fours": 5,
        "sitting on the ground": 6,
    }

    def __init__(self, dataset_root: Path, raw_file_path: Path, reduce: str = "average"):
        self.dataset_root = dataset_root
        self.raw_file_path = raw_file_path
        self.reduce = reduce

    def load(self) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """优先从缓存读取，否则重新解析原始文件。"""
        cache_path = get_records_cache_path(self.dataset_root)
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu")

        records = self._parse_raw_file()
        torch.save(records, cache_path)
        return records

    def _parse_raw_file(self) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """逐行解析活动记录，并把同一时刻/同一传感器的值聚合。"""
        records: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        def save_record(
            record_id: str,
            tt: list[torch.Tensor],
            vals: list[torch.Tensor],
            mask: list[torch.Tensor],
            labels: list[torch.Tensor],
        ) -> None:
            """把当前 record 累积的时间序列整理成最终张量。"""
            tt_tensor = torch.tensor(tt, dtype=torch.float32)
            vals_tensor = torch.stack(vals).reshape(len(vals), -1)
            mask_tensor = torch.stack(mask).reshape(len(mask), -1)
            _ = torch.stack(labels)  # kept to mirror APN-main's parsing flow
            records.append((record_id, tt_tensor, vals_tensor, mask_tensor))

        with open(self.raw_file_path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()

        previous_time = -1
        current_record_id: str | None = None
        tt: list[torch.Tensor] = []
        vals: list[torch.Tensor] = []
        mask: list[torch.Tensor] = []
        nobs: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        first_tp: float | None = None

        for line in lines:
            cur_record_id, tag_id, time, _date, val1, val2, val3, label = line.strip().split(",")
            value_vec = torch.tensor((float(val1), float(val2), float(val3)), dtype=torch.float32)
            time_value = float(time)

            if cur_record_id != current_record_id:
                if current_record_id is not None:
                    save_record(current_record_id, tt, vals, mask, labels)

                current_record_id = cur_record_id
                tt = [torch.zeros(1, dtype=torch.float32)]
                vals = [torch.zeros(len(self.tag_ids), 3, dtype=torch.float32)]
                mask = [torch.zeros(len(self.tag_ids), 3, dtype=torch.float32)]
                nobs = [torch.zeros(len(self.tag_ids), dtype=torch.float32)]
                labels = [torch.zeros(len(self.label_names), dtype=torch.float32)]

                first_tp = time_value
                time_value = round((time_value - first_tp) / 10**4)
                previous_time = time_value
            else:
                if first_tp is None:
                    raise RuntimeError("first_tp must be initialized before continuing a HumanActivity record.")
                time_value = round((time_value - first_tp) / 10**4)

            if time_value != previous_time:
                tt.append(torch.tensor(time_value, dtype=torch.float32))
                vals.append(torch.zeros(len(self.tag_ids), 3, dtype=torch.float32))
                mask.append(torch.zeros(len(self.tag_ids), 3, dtype=torch.float32))
                nobs.append(torch.zeros(len(self.tag_ids), dtype=torch.float32))
                labels.append(torch.zeros(len(self.label_names), dtype=torch.float32))
                previous_time = time_value

            if tag_id not in self.tag_ids:
                if tag_id != "RecordID":
                    raise ValueError(f"Unexpected tag id in HumanActivity data: {tag_id}")
                continue

            tag_index = self.tag_dict[tag_id]
            n_observations = nobs[-1][tag_index]
            if self.reduce == "average" and n_observations > 0:
                previous_value = vals[-1][tag_index]
                new_value = (previous_value * n_observations + value_vec) / (n_observations + 1)
                vals[-1][tag_index] = new_value
            else:
                vals[-1][tag_index] = value_vec

            mask[-1][tag_index] = 1
            nobs[-1][tag_index] += 1

            if label in self.label_names and labels[-1][self.label_dict[label]].sum() == 0:
                labels[-1][self.label_dict[label]] = 1

        if current_record_id is not None:
            save_record(current_record_id, tt, vals, mask, labels)

        return records
