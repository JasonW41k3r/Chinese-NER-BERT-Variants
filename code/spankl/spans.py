from typing import List, Tuple

def tags_to_spans(tags: List[str]) -> List[Tuple[int,int,str]]:
    """
    将连续且相同、且不为 'O' 的标签段合并为 span。
    返回 (start_idx, end_idx_inclusive, label)
    """
    spans = []
    n = len(tags)
    i = 0
    while i < n:
        lab = tags[i]
        if lab == "O":
            i += 1
            continue
        j = i + 1
        while j < n and tags[j] == lab:
            j += 1
        spans.append((i, j-1, lab))
        i = j
    return spans

def greedy_select_non_overlapping(spans_with_scores: List[Tuple[int,int,str,float]]) -> List[Tuple[int,int,str,float]]:
    """
    已按分数降序的候选 spans，贪心选择非重叠集合。
    """
    selected = []
    occupied = []
    for st, ed, lab, sc in spans_with_scores:
        ok = True
        for (ost, oed) in occupied:
            if not (ed < ost or st > oed):
                ok = False
                break
        if ok:
            selected.append((st, ed, lab, sc))
            occupied.append((st, ed))
    return selected

def spans_to_tags(n_tokens: int, pred_spans: List[Tuple[int,int,str]]) -> List[str]:
    """
    将 span 还原成 token 级标签（IO风格：span内全部置为该类标签）。
    """
    tags = ["O"] * n_tokens
    for st, ed, lab in pred_spans:
        for i in range(st, ed+1):
            tags[i] = lab
    return tags
