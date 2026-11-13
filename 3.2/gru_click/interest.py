"""Additional observed-session features, without labels or future timestamps."""

import numpy as np
import pandas as pd


def session_statistics(store, rows, kinds):
    classes = np.asarray(store.classes) + 3
    c = len(classes)
    pair = {}
    global_values = {}
    buffers = {
        name: []
        for name in [
            "session_count",
            "session_share",
            "session_views",
            "session_clicks",
            "session_seen",
            "session_candidate_age",
            "recent8_share",
            "recent16_share",
            "viewed_not_clicked",
            "current_last_target",
        ]
    }
    globals_buffer = {
        name: []
        for name in [
            "utc_hour",
            "utc_weekday",
            "weekend",
            "last_gap_bucket",
            "last_event_kind",
            "session_length",
            "session_products",
            "session_unique_products",
            "session_switch_rate",
            "session_left_censored",
        ]
    }
    for start in range(0, len(rows), 256):
        batch = rows[start : start + 256]
        lengths = store.lengths[batch]
        width = store.manifest["max_length"]
        pos = np.arange(width)[None, :]
        valid = pos < lengths[:, None]
        products = np.asarray(store.tokens["products"][batch])
        events = np.asarray(store.tokens["events"][batch])
        valid = valid & (events > 0)
        boundaries = np.asarray(store.tokens["boundaries"][batch])
        starts = (boundaries == 2) & valid
        last_start = np.where(starts, pos, -1).max(1)
        lo = np.maximum(last_start, 0)
        session = valid & (pos >= lo[:, None])
        product_use = (products >= 3) & session
        event_index = np.maximum(events - 1, 0)
        event_kind = kinds[event_index]
        event_kind = np.where(events > 0, event_kind, 0)
        match = (products[:, None, :] == classes[None, :, None]) & session[:, None, :]
        count = match.sum(2)
        views = (match & (event_kind[:, None, :] == 1)).sum(2)
        clicks = (
            match & ((event_kind[:, None, :] == 2) | (event_kind[:, None, :] == 3))
        ).sum(2)
        last = np.where(match, pos[:, None, :], -1).max(2)
        age = np.where(count > 0, lengths[:, None] - 1 - last, 256)
        n_product = product_use.sum(1)
        recent8 = valid & (pos >= lengths[:, None] - 8)
        recent16 = valid & (pos >= lengths[:, None] - 16)
        all_match = products[:, None, :] == classes[None, :, None]
        targets = (event_kind == 3) & product_use
        last_target = np.where(targets, pos, -1).max(1)
        last_target_product = products[
            np.arange(len(batch)), np.maximum(0, last_target)
        ]
        current_last = (classes[None, :] == last_target_product[:, None]) & (
            last_target[:, None] >= 0
        )
        values = dict(
            session_count=np.log1p(count),
            session_share=count / np.maximum(1, n_product[:, None]),
            session_views=np.log1p(views),
            session_clicks=np.log1p(clicks),
            session_seen=(count > 0).astype(float),
            session_candidate_age=np.log1p(age) / np.log1p(256),
            recent8_share=(all_match & recent8[:, None, :]).sum(2)
            / np.minimum(8, lengths)[:, None],
            recent16_share=(all_match & recent16[:, None, :]).sum(2)
            / np.minimum(16, lengths)[:, None],
            viewed_not_clicked=((views > 0) & (clicks == 0)).astype(float),
            current_last_target=current_last.astype(float),
        )
        for name, v in values.items():
            buffers[name].append(v.astype(np.float32))
        last_pos = lengths - 1
        i = np.arange(len(batch))
        hours = np.asarray(store.tokens["hours"][batch])[i, last_pos]
        weekdays = np.asarray(store.tokens["weekdays"][batch])[i, last_pos]
        day = np.where(weekdays > 0, weekdays - 1, -1)
        unique = []
        switch = []
        for j in range(len(batch)):
            visible = products[j, product_use[j]]
            unique.append(len(np.unique(visible)))
            switch.append(
                float(np.mean(visible[1:] != visible[:-1])) if len(visible) > 1 else 0.0
            )
        g = dict(
            utc_hour=np.where(hours > 0, hours - 1, -1),
            utc_weekday=day,
            weekend=((day >= 5) & (day >= 0)).astype(float),
            last_gap_bucket=np.asarray(store.tokens["gaps"][batch])[i, last_pos],
            last_event_kind=event_kind[i, last_pos],
            session_length=np.log1p(session.sum(1)),
            session_products=np.log1p(n_product),
            session_unique_products=np.log1p(unique),
            session_switch_rate=np.array(switch),
            session_left_censored=(last_start < 0).astype(float),
        )
        for name, v in g.items():
            globals_buffer[name].append(np.asarray(v, np.float32))
    for name, parts in buffers.items():
        pair["s_" + name] = np.concatenate(parts).ravel()
    for name, parts in globals_buffer.items():
        global_values["ctx_" + name] = np.repeat(np.concatenate(parts), c)
    result = pd.DataFrame(dict(**pair, **global_values))
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Nonfinite session features")
    return result
