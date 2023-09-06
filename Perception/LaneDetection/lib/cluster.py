import numpy as np
from scipy.spatial import distance


def naive_cluster_nd(emb_list, gap):
    centers = []  # (mean, num)
    cids = []
    for x, y, emb in emb_list:
        min_gap = gap + 1
        min_cid = -1
        for id, (center, num) in enumerate(centers):
            diff = distance.euclidean(emb, center)
            if diff < min_gap:
                min_gap = diff
                min_cid = id
        if min_gap < gap:
            cids.append((x, y, min_cid))
            center, num = centers[min_cid]
            centers[min_cid] = ((center * num + emb) / (num + 1), num + 1)
        else:
            centers.append((emb, 1))
            cids.append((x, y, len(centers) - 1))
    return cids, centers


def collect_nd_embedding_with_position(seg, emb, conf):
    ret = []
    for i in range(seg.shape[0]):  # H
        for j in range(seg.shape[1]):  # W
            if seg[i, j] >= conf:
                ret.append((i, j, emb[:, i, j]))  # Nd
    return ret


def embedding_post(pred, conf, emb_margin=6.0, min_cluster_size=100):
    seg, emb = pred  # [key]
    seg, emb = seg[0][0], emb[0]

    ret = collect_nd_embedding_with_position(seg, emb, conf)
    c = naive_cluster_nd(ret, emb_margin)

    lanes = np.zeros(seg.shape, dtype=np.uint8)
    for x, y, id in c[0]:
        if c[1][id][1] < min_cluster_size:  # Filter small clusters
            continue
        lanes[x][y] = id + 1
    return lanes, c[0]